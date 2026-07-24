import os
import json
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# FIXED FIGURE 1: COMBINED SEED FIGURE (2x2 Grid)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Figure 1: Combined Multi-Seed Figure (Exact File Targets Mode)...")
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Multi-seed training curves: V-MoE and AttnRoute-MoE B', fontsize=11, fontweight='bold')

for row, seed in enumerate([1, 2]):
    for col, exp in enumerate(['E2', 'E4']):
        lab = 'V-MoE' if exp == 'E2' else 'AttnRoute-MoE B'
        color = 'red' if exp == 'E2' else 'green'

        # Define the EXACT file name strings to prevent any cross-over
        if exp == 'E2':
            target_name = f"E2_VMoE_TinyIN_E4k2_seed{seed}_log.json"
        else:
            target_name = f"E4_VersionB_TinyIN_E4k2_T50_seed{seed}_log.json"

        fpath = os.path.join(log_dir, target_name)

        if os.path.exists(fpath):
            with open(fpath) as f:
                h = json.load(f)['history']
            eps = [e['epoch'] for e in h]
            cvs = [e['cv']    for e in h]
            accs= [e['val_acc']*100 for e in h]

            ax = axes[row][col]
            ax2 = ax.twinx()

            ax.plot(eps, cvs,  color=color, lw=2, label='CV')
            ax2.plot(eps, accs, color='steelblue', lw=1.5, ls='--', label='Acc')

            ax.set_title(f'Seed {seed} — {lab}  CV@ep1={cvs[0]:.4f}', fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('CV (Collapse Rate)', color=color, fontsize=8)
            ax2.set_ylabel('Val acc (%)', color='steelblue', fontsize=8)
            ax.axvline(23, color='gray', ls=':', lw=1)
            ax.grid(alpha=0.2)
            print(f"  ✓ Confirmed Panel Grid [{row}][{col}] mapped cleanly to: {target_name}")
        else:
            print(f"  ❌ Error: File not found at {fpath}")

plt.tight_layout()
fig1_path = os.path.join(out_dir, 'appendix_seeds_combined.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 1 successfully generated -> {fig1_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ALL ABLATION METHODS (Seed 0 Trajectories)
# ══════════════════════════════════════════════════════════════════════════════
print("Processing Figure 2: Ablation Trajectories (Seed 0 Exact Targets)...")
fig, ax = plt.subplots(figsize=(9, 4.5))

# Use the exact seed 0 filenames to avoid multi-seed overlaps
ablation_configs = [
    ('E2_VMoE_TinyImageNet_E4k2_log.json',            'red',    'V-MoE',            '-'),
    ('E4_VersionB_TinyImageNet_E4k2_T50_log.json',    'green',  'Version B (T50%)', '-'),
    ('E10_VersionA_TinyIN_E4k2_log.json',             'purple', 'Version A',        '--'),
    ('E10b_RandomPrior_TinyIN_E4k2_T50_log.json', 'orange', 'Random Prior',     '-.'),
]

for fname, color, lab, ls in ablation_configs:
    fpath = os.path.join(log_dir, fname)

    if os.path.exists(fpath):
        with open(fpath) as f:
            h = json.load(f)['history']
        eps = [e['epoch'] for e in h]; cvs = [e['cv'] for e in h]
        ax.plot(eps, cvs, color=color, lw=2, ls=ls, label=f'{lab} CV@ep1={cvs[0]:.4f}')
        ax.scatter([eps[0]], [cvs[0]], color=color, s=50, zorder=5)
        print(f"  ✓ Plotted curve cleanly from: {fname}")
    else:
        print(f"  ⚠️ Warning: File not found at {fpath}")

ax.axvline(23, color='gray', ls=':', lw=1, label='$\\lambda$→0 (ep23)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Expert collapse CV')
ax.set_title('All ablation methods: CV trajectories (seed 0)', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(out_dir, 'appendix_ablations.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 2 successfully generated -> {fig2_path}")
