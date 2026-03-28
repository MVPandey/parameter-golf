"""product of experts: AR model x learned energy correction.

trains a small energy network on the frozen AR model's residual stream
to correct its per-token probabilities. at eval:

    p_final(v | context) ∝ p_AR(v | context) * exp(-E(v, h_context))

the energy network sees the continuous representation h_context and learns
corrections that the discrete softmax head misses.

usage:
    # train energy corrector on frozen baseline model
    python train_energy_correction.py --checkpoint final_model.pt

    # eval with energy correction
    python train_energy_correction.py --checkpoint final_model.pt --eval-only --energy-ckpt energy_net.pt
"""
from __future__ import annotations
import argparse, glob, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EnergyNet(nn.Module):
    """small network that scores (context_repr, candidate_token) pairs.

    takes the frozen AR model's last hidden state h (d_model) and produces
    energy corrections for all vocab tokens simultaneously.

    E(v, h) = -MLP(h) @ embed(v)  (bilinear in projected space)

    this is NOT just another softmax head — it learns in a different
    representation space (d_energy) with a different geometry."""

    def __init__(self, d_model: int, vocab_size: int, d_energy: int = 128, n_layers: int = 2):
        super().__init__()
        self.d_energy = d_energy

        # project AR hidden state to energy space
        layers = []
        d_in = d_model
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(d_in, d_energy), nn.GELU()])
            d_in = d_energy
        layers.append(nn.Linear(d_in, d_energy))
        self.context_proj = nn.Sequential(*layers)

        # separate token embeddings for energy scoring (not shared with AR model)
        self.token_emb = nn.Embedding(vocab_size, d_energy)

        # learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        self._init()

    def _init(self):
        for m in self.context_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.token_emb.weight, std=0.01)

    def forward(self, h: Tensor) -> Tensor:
        """compute energy corrections for all vocab tokens.
        h: (batch, seq, d_model)
        returns: energy logits (batch, seq, vocab) — ADDITIVE corrections to AR logits"""
        z = self.context_proj(h)  # (batch, seq, d_energy)
        temp = torch.exp(self.log_temp).clamp(min=0.01, max=10.0)
        # bilinear scoring: z @ token_emb.T
        return (z @ self.token_emb.weight.T) / temp


def collect_hidden_states(model, tokens, seq_len, batch_size, device):
    """run frozen AR model and collect (hidden_states, ar_logits, targets)."""
    model.eval()
    all_h = []
    all_logits = []
    all_targets = []

    total_tokens = tokens.numel() - 1
    n_seqs = total_tokens // seq_len

    with torch.inference_mode():
        for bs in range(0, n_seqs, batch_size):
            be = min(bs + batch_size, n_seqs)
            actual_bs = be - bs
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            local = tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(actual_bs, seq_len)
            y = local[1:].reshape(actual_bs, seq_len)

            # run model but capture hidden states before the lm_head
            n = len(model.blocks)
            h = model.tok_emb(x)
            h = F.rms_norm(h, (h.size(-1),))
            h = model.smear(h)
            h0 = h.clone()

            enc = model.num_encoder_layers
            skips = []
            for i in range(n):
                h = model.blocks[i](h, h0)
                if i < enc:
                    skips.append(h.clone())
                elif skips:
                    si = i - enc
                    if si < model.skip_weights.shape[0]:
                        h = h + model.skip_weights[si].to(dtype=h.dtype)[None, None, :] * skips.pop()

            h = model.final_norm(h)

            # AR logits
            logits = F.linear(h, model.tok_emb.weight)
            logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

            all_h.append(h.detach().float().cpu())
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_h), torch.cat(all_logits), torch.cat(all_targets)


def train_energy_net(
    h_states: Tensor,
    ar_logits: Tensor,
    targets: Tensor,
    d_model: int,
    vocab_size: int,
    d_energy: int = 128,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_epochs: int = 5,
    device: torch.device = torch.device('cuda'),
):
    """train the energy correction network on frozen AR model outputs."""

    n_seqs = h_states.shape[0]
    seq_len = h_states.shape[1]
    n_train = int(n_seqs * 0.8)
    n_test = n_seqs - n_train

    energy_net = EnergyNet(d_model, vocab_size, d_energy).to(device)
    opt = torch.optim.Adam(energy_net.parameters(), lr=lr)

    n_steps = (n_train * n_epochs) // batch_size
    print(f'training energy net: {sum(p.numel() for p in energy_net.parameters()):,} params')
    print(f'  {n_train} train seqs, {n_test} test seqs, {n_steps} steps')

    # baseline: AR model alone
    with torch.no_grad():
        test_logits = ar_logits[n_train:].to(device)
        test_targets = targets[n_train:].to(device)
        baseline_ce = F.cross_entropy(
            test_logits.reshape(-1, vocab_size),
            test_targets.reshape(-1),
            reduction='mean'
        ).item()
        baseline_bpb = baseline_ce / math.log(2)
    print(f'  baseline AR: CE={baseline_ce:.4f} BPB≈{baseline_bpb:.4f}')

    t0 = time.perf_counter()
    for step in range(n_steps):
        # cosine lr
        progress = step / max(n_steps - 1, 1)
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g['lr'] = cur_lr

        # random batch from train set
        idx = torch.randint(0, n_train, (batch_size,))
        h = h_states[idx].to(device)       # (bs, seq, d_model)
        ar = ar_logits[idx].to(device)      # (bs, seq, vocab)
        tgt = targets[idx].to(device)       # (bs, seq)

        # energy correction
        energy_logits = energy_net(h)       # (bs, seq, vocab)

        # product of experts: combine AR logits with energy correction
        combined = ar + energy_logits

        # CE loss on combined logits
        loss = F.cross_entropy(
            combined.reshape(-1, vocab_size),
            tgt.reshape(-1),
            reduction='mean'
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy_net.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0 or step == n_steps - 1:
            # eval on test set
            energy_net.eval()
            with torch.no_grad():
                test_h = h_states[n_train:].to(device)
                test_ar = ar_logits[n_train:].to(device)
                test_tgt = targets[n_train:].to(device)

                test_energy = energy_net(test_h)
                test_combined = test_ar + test_energy
                test_ce = F.cross_entropy(
                    test_combined.reshape(-1, vocab_size),
                    test_tgt.reshape(-1),
                    reduction='mean'
                ).item()

                # also measure energy-only (no AR)
                energy_only_ce = F.cross_entropy(
                    test_energy.reshape(-1, vocab_size),
                    test_tgt.reshape(-1),
                    reduction='mean'
                ).item()

                temp = torch.exp(energy_net.log_temp).item()

            energy_net.train()
            delta = test_ce - baseline_ce
            print(f'  step {step+1}/{n_steps} loss={loss.item():.4f} '
                  f'test_ce={test_ce:.4f} (delta={delta:+.4f}) '
                  f'energy_only={energy_only_ce:.4f} temp={temp:.3f} '
                  f'lr={cur_lr:.2e} t={time.perf_counter()-t0:.1f}s')

    # final eval
    energy_net.eval()
    with torch.no_grad():
        test_h = h_states[n_train:].to(device)
        test_ar = ar_logits[n_train:].to(device)
        test_tgt = targets[n_train:].to(device)

        # AR only
        ar_ce = F.cross_entropy(test_ar.reshape(-1, vocab_size), test_tgt.reshape(-1)).item()

        # combined
        test_energy = energy_net(test_h)
        combined_ce = F.cross_entropy((test_ar + test_energy).reshape(-1, vocab_size), test_tgt.reshape(-1)).item()

        # energy only
        energy_ce = F.cross_entropy(test_energy.reshape(-1, vocab_size), test_tgt.reshape(-1)).item()

        # with different mixing weights
        print(f'\n=== FINAL RESULTS ===')
        print(f'AR only:      CE={ar_ce:.6f} BPB≈{ar_ce/math.log(2):.6f}')
        print(f'energy only:  CE={energy_ce:.6f} BPB≈{energy_ce/math.log(2):.6f}')
        print(f'AR + energy:  CE={combined_ce:.6f} BPB≈{combined_ce/math.log(2):.6f}')
        print(f'delta:        {combined_ce - ar_ce:+.6f} CE ({(combined_ce - ar_ce)/math.log(2):+.6f} BPB)')

        # sweep mixing weight
        print(f'\nmixing weight sweep (alpha * energy_logits):')
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            mixed = test_ar + alpha * test_energy
            ce = F.cross_entropy(mixed.reshape(-1, vocab_size), test_tgt.reshape(-1)).item()
            delta = ce - ar_ce
            marker = ' <-- best' if abs(delta) < 0.0001 or (alpha > 0 and delta < 0) else ''
            print(f'  alpha={alpha:.1f}: CE={ce:.6f} delta={delta:+.6f}{marker}')

    return energy_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='final_model.pt')
    parser.add_argument('--d-energy', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=500000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load frozen AR model
    import importlib.util
    spec = importlib.util.spec_from_file_location('tg', 'train_gpt.py')
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    a = tg.Hyperparameters()
    model = tg.GPT(
        vocab_size=a.vocab_size, num_layers=a.num_layers, model_dim=a.model_dim,
        num_heads=a.num_heads, num_kv_heads=a.num_kv_heads, mlp_mult=a.mlp_mult,
        tie_embeddings=a.tie_embeddings, tied_embed_init_std=a.tied_embed_init_std,
        logit_softcap=a.logit_softcap, rope_base=a.rope_base, qk_gain_init=a.qk_gain_init,
    ).to(device).bfloat16()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
    model.eval()
    print(f'loaded AR model: {sum(p.numel() for p in model.parameters()):,} params')

    # load tokens
    val_files = sorted(glob.glob(os.path.join(a.data_path, 'fineweb_val_*.bin')))
    tokens_list = []
    for f in val_files:
        h = np.fromfile(f, dtype='<i4', count=256)
        t = np.fromfile(f, dtype='<u2', count=int(h[2]), offset=256*4)
        tokens_list.append(torch.from_numpy(t.astype(np.uint16, copy=False)))
    tokens = torch.cat(tokens_list)[:args.max_tokens + 1]
    print(f'loaded {tokens.numel():,} tokens')

    # collect hidden states from frozen model
    print('collecting hidden states...')
    t0 = time.perf_counter()
    h_states, ar_logits, targets = collect_hidden_states(
        model, tokens, seq_len=a.train_seq_len if hasattr(a, 'train_seq_len') else 2048,
        batch_size=4, device=device)
    print(f'collected {h_states.shape[0]} sequences x {h_states.shape[1]} tokens in {time.perf_counter()-t0:.1f}s')
    print(f'h_states: {h_states.shape}, ar_logits: {ar_logits.shape}')

    # free the AR model from GPU
    del model
    torch.cuda.empty_cache()

    # train energy correction
    energy_net = train_energy_net(
        h_states, ar_logits, targets,
        d_model=a.model_dim, vocab_size=a.vocab_size,
        d_energy=args.d_energy, batch_size=args.batch_size,
        lr=args.lr, n_epochs=args.n_epochs, device=device,
    )

    torch.save(energy_net.state_dict(), 'energy_net.pt')
    print(f'\nenergy net saved to energy_net.pt ({sum(p.numel() for p in energy_net.parameters()):,} params)')


if __name__ == '__main__':
    main()
