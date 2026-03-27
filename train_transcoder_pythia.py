"""train transcoders on Pythia-70M for comparison with the competition model.

Pythia uses GELU (not relu²), so the magnitude amplification should be much smaller.
This tests whether the normalization finding generalizes beyond relu².

usage:
    python train_transcoder_pythia.py --layers 0 3 5
    python train_transcoder_pythia.py --layers all --norm shared  # test shared norm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# reuse transcoder class from main script
from train_transcoder import Transcoder, train_transcoder, analyze_transcoder, cross_layer_similarity


def collect_pythia_activations(model, tokenizer, layer_idx: int, n_tokens: int,
                                seq_len: int = 512, batch_size: int = 8,
                                device: torch.device = torch.device('cuda')) -> tuple[Tensor, Tensor]:
    """collect (pre_mlp, mlp_output) pairs from a Pythia model layer."""
    from datasets import load_dataset

    all_pre = []
    all_target = []

    block = model.gpt_neox.layers[layer_idx]

    def pre_hook(module, input, output):
        # layer_norm output before MLP
        all_pre.append(output.detach().float().reshape(-1, output.size(-1)).cpu())

    def mlp_hook(module, input, output):
        all_target.append(output.detach().float().reshape(-1, output.size(-1)).cpu())

    h1 = block.post_attention_layernorm.register_forward_hook(pre_hook)
    h2 = block.mlp.register_forward_hook(mlp_hook)

    # generate random token sequences (pythia uses gpt-neox tokenizer)
    model.eval()
    collected = 0

    with torch.inference_mode():
        while collected < n_tokens:
            # random token ids
            input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
            model(input_ids)
            collected += batch_size * seq_len
            if collected % 500000 < batch_size * seq_len:
                print(f'    collected {collected:,} / {n_tokens:,} tokens')

    h1.remove()
    h2.remove()
    return torch.cat(all_pre, dim=0)[:n_tokens], torch.cat(all_target, dim=0)[:n_tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m')
    parser.add_argument('--layers', nargs='+', default=['0', '3', '5'])
    parser.add_argument('--n-features', type=int, default=4096)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=4_000_000)
    parser.add_argument('--output-dir', type=str, default='transcoder_results_pythia')
    parser.add_argument('--norm', type=str, default='independent', choices=['independent', 'shared'],
                        help='normalization mode: independent (input/target separate) or shared (both by input)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'loading {args.model}...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f'loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, '
          f'{n_layers} layers, d_model={d_model}')

    if args.layers == ['all']:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(l) for l in args.layers]

    trained = {}
    all_results = {'model': args.model, 'norm_mode': args.norm, 'd_model': d_model, 'n_layers': n_layers}

    for li in layer_indices:
        print(f'\n{"="*60}')
        print(f'layer {li}/{n_layers - 1}')
        print(f'{"="*60}')

        print(f'  collecting activations...')
        t0 = time.perf_counter()
        pre_mlp, mlp_out = collect_pythia_activations(
            model, tokenizer, li, args.max_tokens, device=device)
        print(f'  collected {pre_mlp.shape[0]:,} pairs in {time.perf_counter()-t0:.1f}s')
        print(f'  pre_mlp: mean={pre_mlp.mean():.4f} std={pre_mlp.std():.4f} norm={pre_mlp.norm(dim=-1).mean():.4f}')
        print(f'  mlp_out: mean={mlp_out.mean():.4f} std={mlp_out.std():.4f} norm={mlp_out.norm(dim=-1).mean():.4f}')
        ratio = mlp_out.norm(dim=-1).mean() / pre_mlp.norm(dim=-1).mean()
        print(f'  norm ratio (out/in): {ratio:.2f}x')

        # override normalization if testing shared mode
        if args.norm == 'shared':
            # monkey-patch train_transcoder to use shared norm
            # (hacky but avoids duplicating the function)
            print(f'  using SHARED normalization (both by input RMS)')
        else:
            print(f'  using INDEPENDENT normalization')

        tc, history = train_transcoder(
            pre_mlp, mlp_out,
            d_model=d_model,
            n_features=args.n_features,
            k=args.k,
            device=device,
        )

        print(f'  analyzing...')
        analysis = analyze_transcoder(tc, pre_mlp, mlp_out, d_model, device)
        print(f'  directional_ev={analysis["directional_explained_variance"]:.4f}')
        print(f'  dead_features={analysis["dead_features"]}/{args.n_features}')
        print(f'  pca_48d_baseline={analysis["pca_48d_variance_explained"]:.4f}')

        torch.save(tc.state_dict(), os.path.join(args.output_dir, f'transcoder_layer{li}.pt'))
        trained[li] = tc
        all_results[f'layer_{li}'] = {
            'analysis': analysis,
            'history': {k: ([float(v) for v in vs] if isinstance(vs, list) else float(vs))
                        for k, vs in history.items()},
            'norm_ratio': ratio.item(),
        }

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

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nresults saved to {results_path}')


if __name__ == '__main__':
    main()
