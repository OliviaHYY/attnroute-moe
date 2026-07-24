# Figure 3 + Figure 4
"""
Generate Figure 3 (expert counts at epoch 1) and Figure 4 (ablation CV curves).
Modded to dynamically route files and paths directly from Google Drive.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── LOAD LOGS WITH COMPLETE PATH ROUTING ─────
def load(fname):
    path = os.path.join(DRIVE_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Log file not found at: {path}. Check your Drive folder!")
    with open(path) as f:
        return json.load(f)['history']

# Update names if needed to match what is printed out by your execution cycles
e2  = load('E2_VMoE_TinyImageNet_E4k2_log.json')
e4  = load('E4_VersionB_TinyImageNet_E4k2_T50_log.json')
e10 = load('E10_VersionA_TinyIN_E4k2_log.json')

# ── FIGURE 3 — Expert counts at epoch 1 ─────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    'Expert token dispatch counts at epoch 1 (before any gradient update)\n'
    'Shows the initial collapse pattern the attention prior is designed to address',
    fontsize=10
)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
x      = np.arange(4)
xlabels = ['E0', 'E1', 'E2', 'E3']

vmoe_ep1 = e2[0]['expert_counts']
vb_ep1   = e4[0]['expert_counts']

for ax, counts, title, cv in [
    (axes[0], vmoe_ep1, 'V-MoE',             0.1654),
    (axes[1], vb_ep1,   'AttnRoute-MoE B',   0.1253),
]:
    bars = ax.bar(x, counts, color=colors, width=0.6, zorder=3)
    mean = np.mean(counts)
    ax.axhline(mean, color='black', ls='--', lw=1.5,
               label=f'Mean = {mean/1e6:.2f}M', zorder=4)
    ratio = min(counts)/max(counts)
    ax.set_title(f'{title}  (CV = {cv:.3f},  min/max = {ratio:.3f})', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel('Token dispatches'); ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1e6:.2f}M'))
    ax.grid(axis='y', alpha=0.3, zorder=0)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000,
                f'{cnt/1e6:.3f}M', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out3 = os.path.join(DRIVE_DIR, 'figure3_expert_counts_ep1.png')  # Saved directly to Drive
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'✨ Saved Figure 3 to Google Drive: {out3}')

# ── FIGURE 4 — Ablation CV trajectories ─────
fig, ax = plt.subplots(figsize=(9, 4.5))

def plot_cv(h, color, label, ls='-', lw=2):
    eps = [e['epoch'] for e in h]
    cvs = [e['cv']    for e in h]
    ax.plot(eps, cvs, color=color, ls=ls, lw=lw, label=label)
    ax.scatter([eps[0]], [cvs[0]], color=color, s=60, zorder=5)
    ax.annotate(f'CV@ep1={cvs[0]:.4f}',
                xy=(eps[0], cvs[0]),
                xytext=(eps[0]+1.5, cvs[0]+0.005),
                fontsize=7.5, color=color)

plot_cv(e2,  '#e41a1c', f'V-MoE (CV@ep1={e2[0]["cv"]:.4f})')
plot_cv(e4,  '#4daf4a', f'AttnRoute-MoE B T50% (CV@ep1={e4[0]["cv"]:.4f})')
plot_cv(e10, '#984ea3', f'Version A — prior only (CV@ep1={e10[0]["cv"]:.4f})',
        ls='--')

ax.axvline(23, color='#4daf4a', ls=':', lw=1.2, alpha=0.7)
ax.text(23.5, 0.157, 'λ→0\n(ep23)', fontsize=7.5, color='#4daf4a', va='top')

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Expert collapse CV', fontsize=11)
ax.set_title(
    'CV trajectories: V-MoE vs AttnRoute-MoE B vs Version A\n'
    'Version A starts lowest but rises; Version B starts low and stabilizes',
    fontsize=10
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()

out4 = os.path.join(DRIVE_DIR, 'figure4_ablation_cv.png')  # Saved directly to Drive
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
print(f' Saved Figure 4 to Google Drive: {out4}')
