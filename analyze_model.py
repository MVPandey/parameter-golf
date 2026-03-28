"""mechanistic interpretability analysis of a parameter golf model.

analyzes why key techniques (BigramHash, SmearGate, XSA, relu²) work
using direct parameter inspection, logit lens, per-layer ablation,
and transcoder feature analysis.

usage:
    python analyze_model.py --checkpoint final_model.pt
"""
from __future__ import annotations
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

def load_model_and_data(checkpoint_path, device):
    """load the competition model and validation tokens."""
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
    missing, unexpected = model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    if missing: print(f'  warning: {len(missing)} missing keys')
    model.eval()

    # load validation tokens
    val_files = sorted(glob.glob(os.path.join(a.data_path, 'fineweb_val_*.bin')))
    tokens = []
    for f in val_files:
        h = np.fromfile(f, dtype='<i4', count=256)
        t = np.fromfile(f, dtype='<u2', count=int(h[2]), offset=256*4)
        tokens.append(torch.from_numpy(t.astype(np.uint16, copy=False)))
    tokens = torch.cat(tokens)

    # load tokenizer for decoding
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer_path)

    return model, tokens, a, sp


def analyze_smeargate(model):
    """inspect the SmearGate's learned mixing weights."""
    print('\n' + '='*60)
    print('SMEARGATE ANALYSIS')
    print('='*60)
    gate = torch.sigmoid(model.smear.gate).detach().cpu()
    print(f'gate range: [{gate.min():.4f}, {gate.max():.4f}]')
    print(f'gate mean: {gate.mean():.4f}')
    print(f'gate std: {gate.std():.4f}')

    # which dimensions mix most with previous token?
    top_k = 10
    vals, idxs = gate.topk(top_k)
    print(f'\ntop {top_k} dimensions mixing with previous token:')
    for i, (v, idx) in enumerate(zip(vals, idxs)):
        print(f'  dim {idx.item()}: gate={v.item():.4f} ({v.item()*100:.1f}% prev token)')

    # which dimensions are mostly current token?
    vals_low, idxs_low = gate.topk(top_k, largest=False)
    print(f'\ntop {top_k} dimensions keeping current token:')
    for i, (v, idx) in enumerate(zip(vals_low, idxs_low)):
        print(f'  dim {idx.item()}: gate={v.item():.4f} ({(1-v.item())*100:.1f}% current token)')

    # distribution
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = torch.histogram(gate, bins=torch.tensor(bins))
    print(f'\ngate distribution:')
    for i in range(len(bins)-1):
        count = int(hist.hist[i].item())
        bar = '#' * (count // 5)
        print(f'  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count:3d} {bar}')

    return {'mean': gate.mean().item(), 'std': gate.std().item(),
            'min': gate.min().item(), 'max': gate.max().item()}


def analyze_bigramhash(model, tokens, sp, device):
    """analyze which bigram patterns the model finds useful."""
    print('\n' + '='*60)
    print('BIGRAMHASH ANALYSIS')
    print('='*60)

    if model.bigram is None:
        print('no bigram module found')
        return {}

    # get bigram embeddings and their norms
    emb = model.bigram.embed.weight.detach().cpu()
    scale = model.bigram.scale.detach().cpu().item()
    print(f'bigram scale: {scale:.4f}')
    print(f'bigram vocab: {emb.shape[0]}, dim: {emb.shape[1]}')

    norms = emb.norm(dim=-1)
    print(f'embedding norms: mean={norms.mean():.4f}, max={norms.max():.4f}')

    # find most activated bigrams on real data
    sample = tokens[:100000].to(torch.int32)
    mod = emb.shape[0] - 1
    hashes = torch.empty_like(sample)
    hashes[0] = mod
    hashes[1:] = torch.bitwise_xor(36313 * sample[1:], 27191 * sample[:-1]) % mod

    # count hash bucket activations
    counts = torch.zeros(emb.shape[0], dtype=torch.long)
    for h in hashes:
        counts[h.item()] += 1

    # top activated buckets
    top_k = 20
    vals, idxs = counts.topk(top_k)
    print(f'\ntop {top_k} most activated bigram buckets:')
    for v, idx in zip(vals, idxs):
        norm = norms[idx].item()
        print(f'  bucket {idx.item()}: count={v.item()}, emb_norm={norm:.4f}')

    # correlation between activation frequency and embedding norm
    mask = counts > 0
    if mask.sum() > 10:
        freq = counts[mask].float()
        n = norms[mask]
        corr = torch.corrcoef(torch.stack([freq, n]))[0, 1].item()
        print(f'\ncorrelation (activation freq vs embedding norm): {corr:.4f}')

    return {'scale': scale, 'n_active_buckets': int(mask.sum().item()),
            'freq_norm_correlation': corr if mask.sum() > 10 else 0.0}


def analyze_logit_lens(model, tokens, sp, device):
    """project each layer's output to vocab and measure prediction quality."""
    print('\n' + '='*60)
    print('LOGIT LENS (when does the model know the answer?)')
    print('='*60)

    n_layers = len(model.blocks)
    sample_tokens = tokens[:2049].to(device=device, dtype=torch.int64)
    x_in = sample_tokens[:-1].unsqueeze(0)  # (1, 2048)
    targets = sample_tokens[1:]  # (2048,)

    # run forward and capture residual stream at each layer
    layer_logits = []

    with torch.inference_mode():
        x = model.tok_emb(x_in)
        x = F.rms_norm(x, (x.size(-1),))
        x = model.smear(x)
        x0 = x.clone()

        enc = model.num_encoder_layers
        skips = []

        for i in range(n_layers):
            x = model.blocks[i](x, x0)
            if i < enc:
                skips.append(x.clone())
            elif skips:
                si = i - enc
                if si < model.skip_weights.shape[0]:
                    x_with_skip = x + model.skip_weights[si].to(dtype=x.dtype)[None, None, :] * skips.pop()
                else:
                    x_with_skip = x

            # project to vocab via logit lens
            h = F.rms_norm(x, (x.size(-1),))
            logits = F.linear(h.squeeze(0), model.tok_emb.weight)
            logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
            layer_logits.append(logits)

    # measure top-1 accuracy and CE loss at each layer
    print(f'\nlayer | top1_acc | top5_acc | CE_loss | entropy')
    print('-' * 60)
    for i, logits in enumerate(layer_logits):
        probs = F.softmax(logits.float(), dim=-1)
        ce = F.cross_entropy(logits.float(), targets, reduction='mean').item()
        top1 = (logits.argmax(dim=-1) == targets).float().mean().item()
        top5 = sum(targets[j] in logits[j].topk(5).indices for j in range(len(targets))) / len(targets)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1).mean().item()
        print(f'  L{i:2d}  | {top1:6.3f}  | {top5:6.3f}  | {ce:6.3f} | {entropy:6.3f}')

    return {'n_layers': n_layers}


def analyze_layer_ablation(model, tokens, device):
    """zero out each layer's MLP contribution and measure CE impact."""
    print('\n' + '='*60)
    print('PER-LAYER MLP ABLATION (which layers matter most?)')
    print('='*60)

    n_layers = len(model.blocks)
    sample = tokens[:4097].to(device=device, dtype=torch.int64)
    x_in = sample[:-1].unsqueeze(0)
    targets = sample[1:].unsqueeze(0)

    # baseline CE
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            baseline_loss = model(x_in, targets).item()
    print(f'baseline CE: {baseline_loss:.4f}')

    # ablate each layer's MLP
    print(f'\nlayer | CE_loss | delta   | importance')
    print('-' * 55)
    deltas = []
    for li in range(n_layers):
        # temporarily zero out mlp_scale for this layer
        orig_scale = model.blocks[li].mlp_scale.data.clone()
        model.blocks[li].mlp_scale.data.zero_()

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ablated_loss = model(x_in, targets).item()

        model.blocks[li].mlp_scale.data.copy_(orig_scale)

        delta = ablated_loss - baseline_loss
        deltas.append(delta)
        bar = '#' * int(delta * 20)
        print(f'  L{li:2d}  | {ablated_loss:6.3f} | +{delta:5.3f} | {bar}')

    # also ablate attention
    print(f'\nlayer | CE_loss | delta   | importance (attention)')
    print('-' * 55)
    for li in range(n_layers):
        orig_scale = model.blocks[li].attn_scale.data.clone()
        model.blocks[li].attn_scale.data.zero_()

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ablated_loss = model(x_in, targets).item()

        model.blocks[li].attn_scale.data.copy_(orig_scale)
        delta = ablated_loss - baseline_loss
        bar = '#' * int(delta * 20)
        print(f'  L{li:2d}  | {ablated_loss:6.3f} | +{delta:5.3f} | {bar}')

    return {'baseline_ce': baseline_loss, 'mlp_deltas': deltas}


def analyze_transcoder_features(model, tokens, sp, device, tc_dir='transcoder_results'):
    """analyze the top features from trained transcoders."""
    print('\n' + '='*60)
    print('TRANSCODER FEATURE ANALYSIS')
    print('='*60)

    from train_transcoder import Transcoder, collect_mlp_activations

    d = 512
    n_features = 4096
    k = 128

    for li in [0, 4, 8]:
        tc_path = os.path.join(tc_dir, f'transcoder_layer{li}.pt')
        if not os.path.exists(tc_path):
            print(f'  layer {li}: no transcoder found at {tc_path}')
            continue

        print(f'\n--- layer {li} ---')
        tc = Transcoder(d, n_features, k).to(device)
        tc.load_state_dict(torch.load(tc_path, map_location=device))
        tc.eval()

        # collect activations
        sample_tokens = tokens[:200001]
        pre, out = collect_mlp_activations(model, li, sample_tokens, 2048, 4, device)

        # normalize
        rms_in = pre.norm(dim=-1, keepdim=True) / math.sqrt(d)
        rms_out = out.norm(dim=-1, keepdim=True) / math.sqrt(d)
        pn = pre / (rms_in + 1e-6)

        # get feature activations
        with torch.no_grad():
            sparse = tc.encode(pn[:50000].to(device))

        # feature frequency
        freq = (sparse > 0).float().mean(dim=0).cpu()
        active = (freq > 1e-6).sum().item()
        print(f'  active features: {active}/{n_features}')

        # top features by activation frequency
        top_freq, top_idx = freq.topk(10)
        print(f'\n  top 10 features by frequency:')
        for rank, (f_val, f_idx) in enumerate(zip(top_freq, top_idx)):
            # find max-activating tokens for this feature
            feat_acts = sparse[:, f_idx.item()].cpu()
            top_acts, top_tok_idx = feat_acts.topk(min(5, (feat_acts > 0).sum().item()))

            # decode the context around max-activating positions
            contexts = []
            for ti in top_tok_idx:
                pos = ti.item()
                # map back to token position (pos in flattened activations)
                seq_idx = pos // 2048
                tok_idx = pos % 2048
                start = max(0, seq_idx * 2048 + tok_idx - 3)
                end = min(sample_tokens.numel(), seq_idx * 2048 + tok_idx + 2)
                ctx_tokens = sample_tokens[start:end].tolist()
                ctx_str = sp.decode(ctx_tokens)
                contexts.append(ctx_str[:60])

            print(f'    F{f_idx.item():4d} freq={f_val.item():.4f} | {contexts[0] if contexts else "?"}')

        # feature decoder norms (which features have largest output impact)
        dec_norms = tc.decoder.weight.data.norm(dim=0).cpu()
        top_impact, top_impact_idx = dec_norms.topk(10)
        print(f'\n  top 10 features by decoder norm (output impact):')
        for v, idx in zip(top_impact, top_impact_idx):
            f = freq[idx.item()].item()
            print(f'    F{idx.item():4d} dec_norm={v.item():.4f} freq={f:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='final_model.pt')
    parser.add_argument('--output', type=str, default='analysis_report.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading model and data...')
    model, tokens, hparams, sp = load_model_and_data(args.checkpoint, device)
    print(f'model: {sum(p.numel() for p in model.parameters()):,} params, {len(model.blocks)} layers')

    results = {}

    # tier 1: direct parameter inspection
    results['smeargate'] = analyze_smeargate(model)
    results['bigramhash'] = analyze_bigramhash(model, tokens, sp, device)

    # tier 2: activation analysis
    results['logit_lens'] = analyze_logit_lens(model, tokens, sp, device)
    results['layer_ablation'] = analyze_layer_ablation(model, tokens, device)

    # tier 3: transcoder features
    analyze_transcoder_features(model, tokens, sp, device)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nresults saved to {args.output}')


if __name__ == '__main__':
    main()
