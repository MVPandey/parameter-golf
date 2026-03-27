"""train transcoders on a pretrained language model to decompose what each MLP layer does.

a transcoder maps pre-MLP residual stream to the MLP's output contribution through a
sparse TopK bottleneck, decomposing the MLP computation into interpretable features.

usage:
    # train transcoders on layers 0, 4, 8 of a trained checkpoint
    python train_transcoder.py --checkpoint final_model.pt --layers 0 4 8

    # train on all layers
    python train_transcoder.py --checkpoint final_model.pt --layers all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# =============================================================================
# transcoder
# =============================================================================

class Transcoder(nn.Module):
    """sparse transcoder: decomposes an MLP's computation into interpretable features.

    maps pre_mlp (d_model) -> sparse features (n_features, topk) -> mlp_output (d_model).
    uses TopK activation instead of L1 to avoid sparsity tuning."""

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)

        # init: encoder rows as random unit vectors, decoder columns as their transposes
        nn.init.xavier_uniform_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        # track feature activation counts for dead feature detection
        self.register_buffer('feature_counts', torch.zeros(n_features, dtype=torch.long))
        self.register_buffer('total_tokens', torch.tensor(0, dtype=torch.long))

    def encode(self, x: Tensor) -> Tensor:
        """encode and apply topk sparsity. returns sparse activations."""
        h = self.encoder(x)  # (batch, n_features)
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        return F.relu(h) * mask

    def decode(self, sparse: Tensor) -> Tensor:
        return self.decoder(sparse)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """returns (reconstruction, sparse_activations)."""
        sparse = self.encode(x)
        recon = self.decode(sparse)
        return recon, sparse

    def aux_loss(self, x: Tensor, target: Tensor, sparse: Tensor) -> Tensor:
        """auxiliary loss on features that didn't make the topk cut.
        trains the 'next-k' features to also reconstruct the mlp output,
        preventing dead features."""
        h = self.encoder(x)
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        aux_h = h.masked_fill(mask, float('-inf'))
        aux_topk_vals, aux_topk_idx = aux_h.topk(self.k, dim=-1)
        aux_sparse = torch.zeros_like(h)
        aux_sparse.scatter_(-1, aux_topk_idx, F.relu(aux_topk_vals))
        aux_recon = self.decode(aux_sparse)
        return F.mse_loss(aux_recon, target.detach())

    @torch.no_grad()
    def update_feature_counts(self, sparse: Tensor):
        active = (sparse > 0).sum(dim=0)
        self.feature_counts += active.long()
        self.total_tokens += sparse.shape[0]

    @torch.no_grad()
    def dead_feature_fraction(self, window_tokens: int = 10_000_000) -> float:
        if self.total_tokens == 0:
            return 0.0
        # features that have fired less than once per window_tokens
        threshold = max(1, self.total_tokens.item() // window_tokens)
        dead = (self.feature_counts < threshold).sum().item()
        return dead / self.n_features

    @torch.no_grad()
    def normalize_decoder(self):
        """constrain decoder columns to unit norm."""
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data /= norms


# =============================================================================
# activation collection from trained model
# =============================================================================

def collect_mlp_activations(model, layer_idx: int, tokens: Tensor, seq_len: int,
                            batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """collect (pre_mlp, mlp_output) pairs from a specific layer using forward hooks.
    works with the upstream GPT model where each block owns its own MLP."""
    block = model.blocks[layer_idx]
    all_pre = []
    all_target = []

    # hook the mlp_norm (captures pre-mlp input after normalization)
    # and the mlp itself (captures mlp output)
    def pre_mlp_hook(module, input, output):
        all_pre.append(output.detach().float().reshape(-1, output.size(-1)).cpu())

    def mlp_hook(module, input, output):
        all_target.append(output.detach().float().reshape(-1, output.size(-1)).cpu())

    h1 = block.mlp_norm.register_forward_hook(pre_mlp_hook)
    h2 = block.mlp.register_forward_hook(mlp_hook)

    model.eval()
    total_tokens = tokens.numel() - 1
    n_seqs = total_tokens // seq_len

    with torch.inference_mode():
        for batch_start in range(0, n_seqs, batch_size):
            batch_end = min(batch_start + batch_size, n_seqs)
            actual_bs = batch_end - batch_start
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x_batch = local[:-1].reshape(actual_bs, seq_len)
            y_batch = local[1:].reshape(actual_bs, seq_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(x_batch, y_batch)

    h1.remove()
    h2.remove()
    return torch.cat(all_pre, dim=0), torch.cat(all_target, dim=0)


# =============================================================================
# training loop
# =============================================================================

def train_transcoder(
    pre_mlp: Tensor,
    mlp_output: Tensor,
    d_model: int = 512,
    n_features: int = 4096,
    k: int = 48,
    batch_size: int = 4096,
    lr: float = 3e-4,
    lr_min: float = 3e-5,
    warmup_steps: int = 1000,
    aux_coeff: float = 1 / 32,
    n_epochs: int = 10,
    device: torch.device = torch.device('cuda'),
    log_every: int = 100,
    test_frac: float = 0.2,
) -> tuple[Transcoder, dict]:
    """train a single transcoder on collected activations."""

    # rms normalize input and target independently
    # directional decomposition: the transcoder learns to predict the direction
    # of the mlp output. magnitude prediction is tracked separately.
    rms_in = pre_mlp.norm(dim=-1, keepdim=True) / math.sqrt(d_model)
    rms_out = mlp_output.norm(dim=-1, keepdim=True) / math.sqrt(d_model)
    pre_norm = pre_mlp / (rms_in + 1e-6)
    target_norm = mlp_output / (rms_out + 1e-6)

    # train/test split to check overfitting
    n_total = pre_norm.shape[0]
    n_test = int(n_total * test_frac)
    n_train = n_total - n_test
    perm = torch.randperm(n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    pre_train, target_train = pre_norm[train_idx], target_norm[train_idx]
    pre_test, target_test = pre_norm[test_idx], target_norm[test_idx]
    # keep raw versions for unnormalized eval
    raw_pre_test = pre_mlp[test_idx]
    raw_target_test = mlp_output[test_idx]
    rms_out_test = rms_out[test_idx]

    n_tokens = pre_train.shape[0]
    n_steps = (n_tokens * n_epochs) // batch_size
    print(f'  training: {n_tokens:,} train, {n_test:,} test, {n_steps:,} steps, {n_epochs} epochs')

    tc = Transcoder(d_model, n_features, k).to(device)
    opt = torch.optim.Adam(tc.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    history = {'mse': [], 'aux': [], 'explained_var': [], 'dead_frac': [], 'step': []}
    t0 = time.perf_counter()

    for step in range(n_steps):
        # lr schedule: warmup then cosine
        if step < warmup_steps:
            cur_lr = lr * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g['lr'] = cur_lr

        # random batch from train split
        idx = torch.randint(0, n_train, (batch_size,))
        x = pre_train[idx].to(device)
        target = target_train[idx].to(device)

        recon, sparse = tc(x)
        mse_loss = F.mse_loss(recon, target)
        aux = tc.aux_loss(x, target, sparse) * aux_coeff
        loss = mse_loss + aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # normalize decoder columns
        tc.normalize_decoder()
        tc.update_feature_counts(sparse.detach())

        if (step + 1) % log_every == 0 or step == 0 or step == n_steps - 1:
            with torch.no_grad():
                var_target = target.var().item()
                ev = 1.0 - mse_loss.item() / max(var_target, 1e-8)
                dead = tc.dead_feature_fraction()

            elapsed = time.perf_counter() - t0
            print(f'  step {step+1}/{n_steps} mse={mse_loss.item():.6f} aux={aux.item():.6f} '
                  f'ev={ev:.4f} dead={dead:.3f} lr={cur_lr:.2e} t={elapsed:.1f}s')

            history['mse'].append(mse_loss.item())
            history['aux'].append(aux.item())
            history['explained_var'].append(ev)
            history['dead_frac'].append(dead)
            history['step'].append(step + 1)

    # evaluate on held-out test set (both normalized and unnormalized)
    tc.eval()
    with torch.no_grad():
        xt = pre_test.to(device)
        tt = target_test.to(device)
        recon_test, _ = tc(xt)
        test_mse_norm = F.mse_loss(recon_test, tt).item()
        test_ev_norm = 1.0 - test_mse_norm / max(tt.var().item(), 1e-8)

        # unnormalized: reconstruct in original scale
        # recon_unnorm = recon_direction * original_output_rms
        recon_unnorm = recon_test.cpu() * rms_out_test
        raw_t = raw_target_test
        unnorm_mse = F.mse_loss(recon_unnorm, raw_t).item()
        unnorm_ev = 1.0 - unnorm_mse / max(raw_t.var().item(), 1e-8)

    print(f'  test (directional): ev={test_ev_norm:.4f} mse={test_mse_norm:.6f}')
    print(f'  test (unnormalized): ev={unnorm_ev:.4f} mse={unnorm_mse:.2f}')
    history['test_ev_directional'] = test_ev_norm
    history['test_ev_unnormalized'] = unnorm_ev
    history['test_mse_directional'] = test_mse_norm
    history['test_mse_unnormalized'] = unnorm_mse

    return tc, history


# =============================================================================
# analysis
# =============================================================================

def analyze_transcoder(tc: Transcoder, pre_mlp: Tensor, mlp_output: Tensor,
                       d_model: int, device: torch.device, n_samples: int = 50000) -> dict:
    """run analysis on a trained transcoder."""
    rms_in = pre_mlp[:n_samples].norm(dim=-1, keepdim=True) / math.sqrt(d_model)
    rms_out = mlp_output[:n_samples].norm(dim=-1, keepdim=True) / math.sqrt(d_model)
    pre_norm = pre_mlp[:n_samples] / (rms_in + 1e-6)
    target_norm = mlp_output[:n_samples] / (rms_out + 1e-6)

    tc.eval()
    with torch.no_grad():
        x = pre_norm.to(device)
        target = target_norm.to(device)
        recon, sparse = tc(x)

        mse = F.mse_loss(recon, target).item()
        var = target.var().item()
        explained_var = 1.0 - mse / max(var, 1e-8)

        # feature activation stats
        active_per_token = (sparse > 0).float().sum(dim=-1).mean().item()
        feature_freq = (sparse > 0).float().mean(dim=0)  # per-feature activation rate
        dead = (feature_freq < 1e-6).sum().item()
        alive = tc.n_features - dead

        # feature activation magnitudes
        active_mask = sparse > 0
        mean_activation = sparse[active_mask].mean().item() if active_mask.any() else 0.0

        # pca baseline: rank-48 linear regression from input to output
        # this measures how well a linear bottleneck approximates the mlp
        n_pca = min(10000, x.shape[0])
        x_c = x[:n_pca] - x[:n_pca].mean(dim=0)
        t_c = target[:n_pca] - target[:n_pca].mean(dim=0)
        from torch.linalg import svd
        U, s, Vh = svd(x_c, full_matrices=False)
        # project input to rank-48, then least-squares fit to target
        x_proj = U[:, :48] * s[:48]  # (n, 48)
        # optimal linear map: (X^T X)^{-1} X^T Y, but simpler via pseudoinverse
        W = torch.linalg.lstsq(x_proj, t_c).solution  # (48, d_model)
        pca_recon = x_proj @ W
        pca_mse = F.mse_loss(pca_recon, t_c).item()
        pca_var_explained = 1.0 - pca_mse / max(t_c.var().item(), 1e-8)

    return {
        'directional_explained_variance': explained_var,
        'directional_mse': mse,
        'target_variance': var,
        'active_features_per_token': active_per_token,
        'dead_features': dead,
        'alive_features': alive,
        'mean_activation_magnitude': mean_activation,
        'feature_freq_min': feature_freq.min().item(),
        'feature_freq_max': feature_freq.max().item(),
        'feature_freq_median': feature_freq.median().item(),
        'pca_48d_variance_explained': float(pca_var_explained),
    }


def cross_layer_similarity(transcoders: dict[int, Transcoder]) -> dict:
    """compute cosine similarity between transcoder encoder weights across layers."""
    layers = sorted(transcoders.keys())
    n = len(layers)
    sim_matrix = {}

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            wi = transcoders[li].encoder.weight.data  # (n_features, d_model)
            wj = transcoders[lj].encoder.weight.data
            # average cosine similarity of top features
            wi_norm = F.normalize(wi, dim=-1)
            wj_norm = F.normalize(wj, dim=-1)
            # max cosine similarity for each feature in li to any feature in lj
            cos_sim = (wi_norm @ wj_norm.T).abs()
            max_sim = cos_sim.max(dim=-1).values.mean().item()
            sim_matrix[f'{li}_{lj}'] = max_sim

    return {'layers': layers, 'similarity': sim_matrix}


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='train transcoders on a pretrained LM')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--layers', nargs='+', default=['0', '4', '8'], help='layer indices or "all"')
    parser.add_argument('--n-features', type=int, default=4096)
    parser.add_argument('--k', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-tokens', type=int, default=4_000_000, help='tokens to collect per layer')
    parser.add_argument('--output-dir', type=str, default='transcoder_results')
    parser.add_argument('--data-path', type=str, default='./data/datasets/fineweb10B_sp1024')
    parser.add_argument('--tokenizer-path', type=str, default='./data/tokenizers/fineweb_1024_bpe.model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    print(f'loading checkpoint: {args.checkpoint}')

    # import model from train_gpt.py (upstream naive baseline: GPT + Hyperparameters)
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_gpt', 'train_gpt.py')
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)

    a = tg.Hyperparameters()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    model = tg.GPT(
        vocab_size=a.vocab_size, num_layers=a.num_layers, model_dim=a.model_dim,
        num_heads=a.num_heads, num_kv_heads=a.num_kv_heads, mlp_mult=a.mlp_mult,
        tie_embeddings=a.tie_embeddings, tied_embed_init_std=a.tied_embed_init_std,
        logit_softcap=a.logit_softcap, rope_base=a.rope_base,
        qk_gain_init=a.qk_gain_init,
    ).to(device).bfloat16()
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f'  warning: missing keys: {missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  warning: unexpected keys: {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
    model.eval()
    print(f'model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    # load validation tokens
    import glob
    val_pattern = os.path.join(args.data_path, 'fineweb_val_*.bin')
    val_files = sorted(glob.glob(val_pattern))
    tokens_list = []
    for f in val_files:
        header = np.fromfile(f, dtype='<i4', count=256)
        t = np.fromfile(f, dtype='<u2', count=int(header[2]), offset=256 * 4)
        tokens_list.append(torch.from_numpy(t.astype(np.uint16, copy=False)))
    tokens = torch.cat(tokens_list).contiguous()
    max_tokens = min(args.max_tokens, tokens.numel() - 1)
    tokens = tokens[:max_tokens + 1]
    print(f'loaded {tokens.numel():,} validation tokens')

    # determine layers
    n_layers = a.num_layers
    if args.layers == ['all']:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(l) for l in args.layers]
    print(f'will train transcoders for layers: {layer_indices}')

    # train transcoders
    trained = {}
    all_results = {}

    for li in layer_indices:
        print(f'\n{"="*60}')
        print(f'layer {li}/{n_layers - 1}')
        print(f'{"="*60}')

        # collect activations
        print(f'  collecting activations...')
        t0 = time.perf_counter()
        pre_mlp, mlp_out = collect_mlp_activations(
            model, li, tokens, seq_len=a.train_seq_len if hasattr(a, 'train_seq_len') else 2048,
            batch_size=8, device=device)
        print(f'  collected {pre_mlp.shape[0]:,} activation pairs in {time.perf_counter()-t0:.1f}s')
        print(f'  pre_mlp: mean={pre_mlp.mean():.4f} std={pre_mlp.std():.4f} norm={pre_mlp.norm(dim=-1).mean():.4f}')
        print(f'  mlp_out: mean={mlp_out.mean():.4f} std={mlp_out.std():.4f} norm={mlp_out.norm(dim=-1).mean():.4f}')

        # train
        tc, history = train_transcoder(
            pre_mlp, mlp_out,
            d_model=a.model_dim,
            n_features=args.n_features,
            k=args.k,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        # analyze
        print(f'  analyzing...')
        analysis = analyze_transcoder(tc, pre_mlp, mlp_out, a.model_dim, device)
        print(f'  directional_ev={analysis["directional_explained_variance"]:.4f}')
        print(f'  dead_features={analysis["dead_features"]}/{args.n_features}')
        print(f'  pca_48d_baseline={analysis["pca_48d_variance_explained"]:.4f}')
        print(f'  active_per_token={analysis["active_features_per_token"]:.1f}')

        # save
        torch.save(tc.state_dict(), os.path.join(args.output_dir, f'transcoder_layer{li}.pt'))
        trained[li] = tc
        all_results[f'layer_{li}'] = {
            'analysis': analysis,
            'history': {k: [float(v) for v in vs] for k, vs in history.items()},
        }

    # cross-layer similarity
    if len(trained) > 1:
        print(f'\n{"="*60}')
        print('cross-layer similarity')
        print(f'{"="*60}')
        sim = cross_layer_similarity(trained)
        all_results['cross_layer_similarity'] = sim
        layers = sim['layers']
        for i in layers:
            row = [f'{sim["similarity"][f"{i}_{j}"]:.3f}' for j in layers]
            print(f'  layer {i}: {" ".join(row)}')

    # save all results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nresults saved to {results_path}')


if __name__ == '__main__':
    main()
