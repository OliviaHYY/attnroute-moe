# AttnRoute-MoE

**Attention-Prior Routing for Mixture-of-Experts Vision Transformers**  
Hanyu (Olivia) Yu

[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21635491-blue)](https://doi.org/10.5281/zenodo.21635491)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, experiment logs, and figures for the AttnRoute-MoE paper.

---

## Overview

AttnRoute-MoE uses three scalar signals from the self-attention matrix
(entropy, cross-head variance, CLS-similarity) as a routing prior that
anneals to zero during training, leaving a standard MoE at inference.
**Key finding:** routing diversity — not semantic content — is the primary
factor in preventing expert collapse early in training.

---

## Repository structure
experiments_logs/ # JSON logs for all experiments (E1–E17)
models/ # Model class definitions (AttnRouteMoEBlock, etc.)
notebooks/ # Kaggle-compatible training notebooks
utils/ # Figure generation, analysis utilities
data/ # Data loading utilities (Tiny ImageNet)


## Setup

```bash
pip install torch torchvision transformers numpy matplotlib
# Tiny ImageNet: download from http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

## Reproducing experiments

| Experiment | Description |
|---|---|---|
| E1 Dense | Dense ViT-B baseline | 
| E2 V-MoE | V-MoE baseline (3 seeds) | 
| E4 Version B | AttnRoute-MoE T50% (3 seeds) 
| E7 Entropy only | Signal dropout ablation | 
| E10b Random Prior | Random prior ablation |
| E17 Specialization | Expert routing analysis |

All training was run on Google Colab (A100 GPU).

## Citation

If you use this code, please cite:

```bibtex
@misc{yu2026attnroute,
  title   = {AttnRoute-MoE: Attention-Prior Routing for
             Mixture-of-Experts Vision Transformers},
  author  = {Yu, Hanyu (Olivia)},
  year    = {2026},
  doi     = {10.5281/zenodo.21635491},
  url     = {https://doi.org/10.5281/zenodo.21635491}
}
```

## License

MIT — see [LICENSE](LICENSE).
