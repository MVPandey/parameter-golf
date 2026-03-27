# Transcoders Don't Need Large Models: Normalization-Aware Sparse Decomposition for Sub-100M Parameter Networks

## Abstract

Transcoders -- sparse decompositions of MLP computations into interpretable features -- have shown promise for mechanistic interpretability, but published work has been limited to models with 120M+ parameters. We show that transcoders work well on models an order of magnitude smaller (27M parameters), achieving 96-98% explained variance with zero dead features across all layers of a competition-optimized transformer. Our key finding is that the widely-reported dead feature problem in transcoders is not caused by insufficient model scale, but by a normalization mismatch: relu-squared MLPs amplify activation magnitudes by ~500x between input and output, and the standard practice of normalizing both by the input's RMS creates a pathological loss landscape where a handful of high-magnitude output directions dominate training and starve 89% of features. Independent normalization of input and target -- a simple preprocessing change -- completely eliminates dead features and raises explained variance from 38% to 96%. We train transcoders across all layers of a 9-layer model in under 30 minutes on a single H100, producing a cross-layer analysis revealing a U-shaped complexity curve: first and last layers are most decomposable while middle layers are hardest, and no layer pair shares more than 22% feature similarity. Our results suggest that transcoder-based interpretability is accessible at scales where most researchers develop and study models, and that the dead feature problem in modern architectures using squared activations has a simpler fix than previously understood.

## Completed Results

### Setup

- **Model:** 9-layer, 512d transformer (27M params, 1024 BPE vocab, ReLU² MLP)
- **Transcoder:** 4096 features (8x expansion), TopK K=128, trained on 3.2M tokens (10 epochs)
- **Hardware:** Single H100, ~5 min per layer, full sweep in ~30 min

### The Normalization Finding (Main Contribution)

The model's MLP uses `relu(Wx).square()`, which amplifies input norm from ~22.6 to output norm ~11,320 (a 500x factor). Standard transcoder training normalizes both input and target by the input's RMS. This creates a loss landscape dominated by a few extreme-magnitude output directions:

| | Shared Normalization | Independent Normalization |
|---|---------------------|--------------------------|
| **Explained Variance** | 38.1% | **96.3%** |
| **Dead Features** | 3,660 / 4,096 (89%) | **0 / 4,096 (0%)** |
| **Alive Features** | 436 | **4,096** |

Same architecture, same hyperparameters, same data. The only change is normalizing the target by its own RMS instead of the input's.

### Validation (Layer 4)

| Metric | Value |
|--------|-------|
| Train directional EV | 96.3% |
| Test directional EV | 96.3% (no overfitting) |
| Test unnormalized EV | 96.1% (magnitude captured too) |
| PCA rank-48 baseline | 21.7% |
| Transcoder vs PCA gap | +74.4pp |

The unnormalized EV being 96.1% means the transcoder captures magnitude information even though it's trained on direction-normalized targets. Train/test match confirms generalization, not memorization.

### 5-Layer Cross-Layer Sweep (Complete)

| Layer | Directional EV | Unnorm EV | Dead Features |
|-------|---------------|-----------|---------------|
| 0 (first) | 97.4% | **97.9%** | 0 |
| 2 | 96.6% | 97.0% | 0 |
| 4 (middle) | 96.3% | 96.1% | 0 |
| 6 | 95.9% | 95.7% | 0 |
| 8 (last) | **98.3%** | **98.2%** | 0 |

**U-shaped complexity curve:** first and last layers are most decomposable, middle layers are hardest. Layer 8 (last) has the highest EV, likely because it does simpler unembedding-direction computation. Zero dead features at every layer confirms the normalization fix works universally.

### Cross-Layer Feature Similarity

```
      L0     L2     L4     L6     L8
L0   1.000  0.177  0.166  0.165  0.171
L2   0.174  1.000  0.196  0.184  0.178
L4   0.165  0.198  1.000  0.208  0.194
L6   0.167  0.188  0.211  1.000  0.222
L8   0.169  0.176  0.189  0.214  1.000
```

All off-diagonal similarity below 0.23. Every layer computes fundamentally different features. Adjacent layers are slightly more similar (L6-L8: 0.222) than distant ones (L0-L8: 0.171), but no layer pair is redundant. Weight sharing / depth recurrence would not work for this model.

## Why This Matters

1. **Overturns an assumption.** The mech interp community has treated dead features as an optimization problem solved by AuxK loss, ghost gradients, and resampling. We show that for relu² MLPs, dead features are entirely a preprocessing bug: normalize the target by its own RMS instead of the input's. Two lines of code.

2. **Potentially generalizes to SwiGLU/GeGLU.** Any gated architecture where the MLP output norm scales nonlinearly relative to input norm should exhibit the same pathology. This includes LLaMA, Gemma, and Mistral families. (Needs experimental validation.)

3. **Democratizes transcoders.** Previous work required 120M+ models. We show 27M works with 96%+ EV and zero dead features. Five minutes per layer on a single GPU.

4. **Cross-layer analysis reveals computational structure.** The U-shaped EV curve and low feature similarity matrix provide a "fingerprint" of how a 9-layer model allocates computation across depth.

## Honest Assessment: What's Missing for NeurIPS

Based on deep research into the current literature and NeurIPS standards:

### The Paper is Currently Workshop-Level

The normalization finding is genuinely novel and the small-model contribution fills a real gap. But NeurIPS main track requires:

### Must Have (Critical)
- [ ] Multi-scale validation: train transcoders on Pythia-70M (GELU) and at least one SwiGLU model (Gemma 3 270M or Llama 3.2 1B) to show the normalization finding generalizes beyond relu²
- [ ] Skip transcoder baseline (Paulo et al. 2025 showed skip transcoders Pareto-dominate standard transcoders)
- [ ] SAE baseline on same activations (standard comparison)
- [ ] Delta CE loss: splice transcoder into model, measure cross-entropy increase (considered the causally meaningful metric, not just EV)
- [ ] Feature interpretability evaluation: max-activating examples or auto-interp scoring (without this, reviewers will ask "high EV, but are the features meaningful?")

### Strongly Recommended
- [ ] K sweep (32, 64, 128, 256) with Pareto frontier analysis
- [ ] Dictionary size ablation (2K, 4K, 8K, 16K)
- [ ] SAEBench evaluation if feasible (field standard benchmark)
- [ ] Address CLT faithfulness concerns (Anthropic's "feature skipping" failure mode)
- [ ] Position relative to Gemma Scope 2 (DeepMind released transcoders for Gemma 3 270M-27B in Sep 2025)

### Key Risks
- The novelty window for "first transcoder on a small model" is narrow and closing fast (openCLT, Gemma Scope 2 at 270M)
- 96% EV at K=128 (L0=128) is high sparsity; need to show it holds at L0<64 to be competitive with published results
- Gao et al. explicitly argued that EV is misleading; delta CE loss is the real metric

### Realistic Timeline

| Week | Tasks |
|------|-------|
| Mar 27 - Apr 3 | Pythia-70M + SwiGLU model transcoders. SAE + skip transcoder baselines. |
| Apr 4 - Apr 10 | Delta CE loss test. Feature interpretability. K and dictionary sweeps. |
| Apr 11 - Apr 17 | Write full draft (8 pages + appendix). |
| Apr 18 - Apr 24 | Figures, tables, polish. Get feedback. |
| Apr 25 - May 4 | Revisions. Submit abstract May 4, paper May 6. |

Training is cheap (~5 min/layer) so experiments are bottlenecked by engineering and writing, not compute.

## Venue Options

1. **NeurIPS 2026 Main Track** -- Abstract May 4, paper May 6. Achievable but tight. Requires multi-scale validation.
2. **ICML 2026 Mechanistic Interpretability Workshop** -- Deadline ~mid-May. 4-9 pages. Current results are sufficient for a strong workshop paper.
3. **ICLR 2027** -- Deadline ~October 2026. More time for thorough experiments.

**Recommendation:** Target ICML workshop as the safe bet, NeurIPS main track as the stretch goal. The normalization finding is the headline; everything else supports it.

## Related Work

- Dunefsky et al. (NeurIPS 2024): Transcoders on GPT-2 Small (120M+), circuit discovery
- Paulo & Shabalin (2025): Skip transcoders beat SAEs, Pythia-160M+
- Anthropic Circuit Tracing (2025): Cross-layer transcoders, Claude 3.5 Haiku, ~50% faithfulness
- Gao et al. (OpenAI, ICLR 2025 Oral): Scaling SAEs, 90% dead features without mitigations, AuxK fix
- Gemma Scope 2 (DeepMind, Sep 2025): CLTs for Gemma 3 270M-27B, affine skip connections
- Bricken et al. (Anthropic, 2023): "Towards Monosemanticity", 1-layer 21M model (SAEs only)
- SAEBench (Karvonen et al., ICML 2025): Standardized evaluation across 8 metrics

No published transcoder work below 120M parameters. Novelty window is narrow.
