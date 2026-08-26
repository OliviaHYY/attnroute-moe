# AttnRoute-MoE — Supplementary Code and Logs

## Setup
pip install torch torchvision transformers numpy matplotlib

## Reproduce main results (E1-E4, E10(b)) (Figure 2 and Table 2)
python training/train_main.py --exp E4 --seed 0
  --data_dir /path/to/tiny-imagenet-200 --out_dir ./outputs

## Reproduce ablations (E7 only)  
python training/train_ablation.py --mode full --exp E7 ...

## Reproduce Figure 3 and 4

## Reproduce Figure 5b
python utils/make_figure5b.py --ckpt_vb <VB_best.pt>
  --ckpt_rp <RP_best.pt> ...

## Experiments logs
All JSON training logs are in experiments/logs/. Each file contains per-epoch
val_acc, CV, lambda, and expert_counts.

Code will be publicly released upon acceptance.
