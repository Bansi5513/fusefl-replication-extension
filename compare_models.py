import os
import re
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))  # automatically uses the folder this script is in

MODELS = {
    "ochiai_sbfl":            "Ochiai (SBFL)",
    "baseline_paper_prompt":  "Baseline\n(Wu et al.)",
    "fusefl_prompt":          "FuseFL\n(GPT-3.5)",
    "groq_llama3_70b_prompt": "Llama 3.3 70B\n(Groq)",
    "groq_llama3_8b_prompt":  "Llama 3.1 8B\n(Groq)",
    "groq_qwen3_prompt":      "Qwen3 32B\n(Groq)",
    "groq_gptoss_prompt":     "GPT-OSS 20B\n(Groq)",
}

FIXED_COLORS = {
    "ochiai_sbfl":           "#5F8DB8",
    "baseline_paper_prompt": "#B85F5F",
    "fusefl_prompt":         "#2E7D32",
}

AUTO_PALETTE = [
    "#E67E22", "#9B59B6", "#1ABC9C", "#E74C3C",
    "#3498DB", "#F39C12", "#D35400", "#16A085",
]

QUESTIONS = ['question_1', 'question_2', 'question_3',
             'question_4', 'question_5']


def load_ground_truth():
    gt   = defaultdict(list)
    path = os.path.join(BASE, "Dataset", "Faulty_Line_Explanation.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth not found: {path}")
    with open(path, encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            try:
                gt[row['File'].strip()].append(int(row['Buggy Line'].strip()))
            except (ValueError, KeyError):
                pass
    return gt


def load_results_line(results_dir):
    preds = {}
    for q in QUESTIONS:
        path = os.path.join(results_dir, q, 'results_line')
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if not fname.endswith('.txt'):
                continue
            key = fname.replace('.txt', '')
            try:
                with open(os.path.join(path, fname),
                          encoding='utf-8', errors='replace') as fh:
                    content = fh.read().strip()
                    if content.startswith("FAILED"):
                        continue
                    nums = [int(x.strip()) for x in content.split(',')
                            if x.strip().isdigit()]
                    if nums:
                        preds[key] = nums
            except Exception:
                pass
    return preds


def load_sbfl_from_prompts():
    preds    = {}
    base_dir = os.path.join(BASE, "Prompts", "FuseFL")
    for q in QUESTIONS:
        qdir = os.path.join(base_dir, q)
        if not os.path.exists(qdir):
            continue
        for fname in os.listdir(qdir):
            if not fname.endswith('.txt'):
                continue
            key = fname.replace('.txt', '')
            try:
                with open(os.path.join(qdir, fname),
                          encoding='utf-8', errors='replace') as fh:
                    matches = re.findall(r'\d+\.\s+Line\s+(\d+)', fh.read())
                    if matches:
                        preds[key] = [int(m) for m in matches]
            except Exception:
                pass
    return preds


def compute_topk(preds, gt, ks=(1, 2, 3)):
    counts  = {k: 0 for k in ks}
    covered = 0
    for fname, buggy in gt.items():
        if fname not in preds:
            continue
        covered += 1
        ranked  = preds[fname]
        for k in ks:
            if any(b in ranked[:k] for b in buggy):
                counts[k] += 1
    return counts, covered


def hit_vector(preds, gt, k=1):
    return [int(any(b in preds.get(f, [])[:k] for b in buggy))
            for f, buggy in gt.items()]


def cohens_d(a, b):
    a, b   = np.array(a, float), np.array(b, float)
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return abs(np.mean(a) - np.mean(b)) / pooled if pooled else 0.0


def effect_label(d):
    if d < 0.2: return "Negligible"
    if d < 0.5: return "Small"
    if d < 0.8: return "Medium"
    return "Large"


def wilcoxon_p(a, b):
    try:
        _, p = stats.wilcoxon(a, b, zero_method='zsplit')
        return p
    except Exception:
        return float('nan')


def assign_colors(keys):
    colors = {}
    idx    = 0
    for k in keys:
        if k in FIXED_COLORS:
            colors[k] = FIXED_COLORS[k]
        else:
            colors[k] = AUTO_PALETTE[idx % len(AUTO_PALETTE)]
            idx += 1
    return colors


def main():
    print("=" * 65)
    print("  FuseFL Multi-Model Comparison")
    print("=" * 65)

    print("\n[1/4] Loading ground truth...")
    gt    = load_ground_truth()
    total = len(gt)
    print(f"      {total} files, "
          f"{sum(len(v) for v in gt.values())} buggy line entries")

    print("\n[2/4] Loading model predictions...")
    all_preds = {}
    available = {}

    for key, label in MODELS.items():
        if key == "ochiai_sbfl":
            preds = load_sbfl_from_prompts()
        else:
            rdir = os.path.join(BASE, "Prompts_Results", key)
            if not os.path.exists(rdir):
                print(f"      [SKIP] {key} — no results folder")
                continue
            preds = load_results_line(rdir)
            if not preds:
                print(f"      [SKIP] {key} — no results yet")
                continue

        all_preds[key] = preds
        available[key] = label
        print(f"      {label.replace(chr(10),' '):<42} {len(preds):>4} files")

    if len(available) < 2:
        print("\n  Need at least 2 models. Run multi_model_runner.py first.")
        return

    print("\n[3/4] Computing scores...")
    scores = {}
    for key in available:
        topk, covered = compute_topk(all_preds[key], gt)
        scores[key]   = {"topk": topk, "covered": covered}

    W = 42
    print()
    print(f"  {'Model':<{W}} {'Top-1':>8}  {'%':>6}  "
          f"{'Top-2':>6}  {'Top-3':>6}  {'Coverage':>9}")
    print("  " + "─" * 80)
    for key, label in available.items():
        s   = scores[key]["topk"]
        cov = scores[key]["covered"]
        pct = s[1] / total * 100 if total else 0
        print(f"  {label.replace(chr(10),' '):<{W}} {s[1]:>6}  "
              f"({pct:4.1f}%)  {s[2]:>6}  {s[3]:>6}  {cov:>5}/{total}")

    fusefl_key = "fusefl_prompt"
    stat_rows  = {}
    if fusefl_key in all_preds:
        fh = hit_vector(all_preds[fusefl_key], gt, k=1)
        print(f"\n  Wilcoxon significance vs FuseFL-GPT at Top-1")
        print(f"  {'Model':<{W}} {'p-value':>10}  {'Sig@5%':>7}  "
              f"{'Cohen d':>8}  Effect")
        print("  " + "─" * 80)
        for key, label in available.items():
            if key == fusefl_key:
                continue
            oh  = hit_vector(all_preds[key], gt, k=1)
            p   = wilcoxon_p(fh, oh)
            d   = cohens_d(fh, oh)
            sig = "YES" if (not np.isnan(p) and p < 0.05) else "no"
            stat_rows[key] = {"p": p, "d": d, "sig": sig}
            p_s = f"{p:.4f}" if not np.isnan(p) else "N/A"
            print(f"  {label.replace(chr(10),' '):<{W}} {p_s:>10}  "
                  f"{sig:>7}  {d:>8.3f}  {effect_label(d)}")

    ochiai_t1 = scores.get("ochiai_sbfl",  {}).get("topk", {}).get(1)
    fusefl_t1 = scores.get(fusefl_key,     {}).get("topk", {}).get(1)

    if ochiai_t1:
        print(f"\n  Improvement at Top-1")
        print(f"  {'Model':<{W}} {'vs Ochiai':>11}  {'vs FuseFL-GPT':>14}")
        print("  " + "─" * 72)
        for key, label in available.items():
            if key == "ochiai_sbfl":
                continue
            t1 = scores[key]["topk"][1]
            vo = f"+{(t1-ochiai_t1)/ochiai_t1*100:.1f}%" if ochiai_t1 else "N/A"
            vf = (f"{(t1-fusefl_t1)/fusefl_t1*100:+.1f}%"
                  if fusefl_t1 else "N/A")
            print(f"  {label.replace(chr(10),' '):<{W}} {vo:>11}  {vf:>14}")

    print("\n[4/4] Saving charts and report...")
    colors = assign_colors(list(available.keys()))
    _save_report(available, scores, stat_rows, gt, total, ochiai_t1, fusefl_t1)
    _chart_topk(available, scores, colors, total)
    if ochiai_t1 and len(available) > 1:
        _chart_improvement(available, scores, colors, ochiai_t1, fusefl_t1)
    if len(available) >= 2:
        _chart_heatmap(available, all_preds, gt)

    print("\n" + "=" * 65)
    print("  Done!")
    for f in ["multi_model_comparison.txt",
              "multi_model_topk.png",
              "multi_model_improvement.png",
              "multi_model_heatmap.png"]:
        if os.path.exists(f):
            print(f"    ✓ {f}")
    print("=" * 65)


def _save_report(available, scores, stat_rows, gt, total,
                 ochiai_t1, fusefl_t1):
    lines = [
        "=" * 65,
        "  FuseFL MULTI-MODEL COMPARISON REPORT",
        "=" * 65,
        f"\nDataset : {total} files | "
        f"{sum(len(v) for v in gt.values())} buggy line entries",
        f"Models  : {len(available)} evaluated\n",
        "TOP-K FAULT LOCALIZATION RESULTS",
        f"  {'Model':<42} {'Top-1':>6}  {'%':>7}  {'Top-2':>6}  {'Top-3':>6}",
        "  " + "─" * 68,
    ]
    for key, label in available.items():
        s   = scores[key]["topk"]
        pct = f"{s[1]/total*100:.1f}%" if total else "N/A"
        lines.append(f"  {label.replace(chr(10),' '):<42} "
                     f"{s[1]:>6}  {pct:>7}  {s[2]:>6}  {s[3]:>6}")

    if ochiai_t1:
        lines += ["", "IMPROVEMENT AT TOP-1",
                  f"  {'Model':<42} {'vs Ochiai':>10}  {'vs FuseFL-GPT':>14}",
                  "  " + "─" * 70]
        for key, label in available.items():
            if key == "ochiai_sbfl":
                continue
            t1 = scores[key]["topk"][1]
            vo = f"+{(t1-ochiai_t1)/ochiai_t1*100:.1f}%" if ochiai_t1 else "N/A"
            vf = (f"{(t1-fusefl_t1)/fusefl_t1*100:+.1f}%"
                  if fusefl_t1 else "N/A")
            lines.append(f"  {label.replace(chr(10),' '):<42} "
                         f"{vo:>10}  {vf:>14}")

    if stat_rows:
        lines += ["", "STATISTICAL SIGNIFICANCE vs FuseFL-GPT (Top-1)",
                  f"  {'Model':<42} {'p-value':>10}  {'Sig@5%':>7}  Effect",
                  "  " + "─" * 70]
        for key, r in stat_rows.items():
            lbl = available[key].replace('\n', ' ')
            p_s = f"{r['p']:.4f}" if not np.isnan(r['p']) else "N/A"
            lines.append(f"  {lbl:<42} {p_s:>10}  {r['sig']:>7}  "
                         f"{effect_label(r['d'])} (d={r['d']:.3f})")

    with open("multi_model_comparison.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("      multi_model_comparison.txt ✓")


def _chart_topk(available, scores, colors, total):
    keys   = list(available.keys())
    labels = [available[k] for k in keys]
    n      = len(keys)
    w_bar  = min(0.55, 4.0 / max(n, 1))

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 2.2), 7))
    fig.suptitle("FuseFL: Fault Localization Performance by Model",
                 fontsize=13, fontweight='bold', y=1.01)

    for i, k in enumerate([1, 2, 3]):
        ax   = axes[i]
        vals = [scores[mk]["topk"][k] for mk in keys]
        x    = np.arange(n)
        bars = ax.bar(x, vals, width=w_bar,
                      color=[colors[mk] for mk in keys],
                      alpha=0.88, edgecolor='white',
                      linewidth=1.2, zorder=3)
        for bar, val in zip(bars, vals):
            pct = f"\n({val/total*100:.0f}%)" if total else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val}{pct}",
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

        ax.set_title(f"Top-{k}", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        # Fixed: rotation + ha to prevent overlap
        ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha='right')
        ax.set_ylabel("Files correctly localized" if i == 0 else "")
        ax.set_ylim(0, max(vals) * 1.28 + 3 if vals else 10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("multi_model_topk.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("      multi_model_topk.png ✓")


def _chart_improvement(available, scores, colors, ochiai_t1, fusefl_t1):
    keys   = [k for k in available if k != "ochiai_sbfl"]
    labels = [available[k] for k in keys]
    n      = len(keys)
    if n == 0:
        return
    x = np.arange(n)
    w = min(0.32, 2.2 / max(n, 1))

    vs_ochiai = [(scores[k]["topk"][1] - ochiai_t1) / ochiai_t1 * 100
                 for k in keys]
    vs_fusefl = ([(scores[k]["topk"][1] - fusefl_t1) / fusefl_t1 * 100
                  for k in keys]
                 if fusefl_t1 else [])

    fig, ax = plt.subplots(figsize=(max(11, n * 2.0), 6))
    clrs    = [colors[k] for k in keys]
    offset  = -w / 2 if vs_fusefl else 0

    b1 = ax.bar(x + offset, vs_ochiai, w,
                color=clrs, alpha=0.88, edgecolor='white',
                linewidth=1.2, label="vs Ochiai (SBFL)", zorder=3)
    if vs_fusefl:
        ax.bar(x + w / 2, vs_fusefl, w,
               color=clrs, alpha=0.40, edgecolor='grey',
               linewidth=1.2, hatch='///',
               label="vs FuseFL (GPT-3.5)", zorder=3)

    for bar in b1:
        h  = bar.get_height()
        yp = h + 0.5 if h >= 0 else h - 2.5
        ax.text(bar.get_x() + bar.get_width() / 2, yp,
                f"{h:+.1f}%", ha='center',
                fontsize=7.5, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.9)
    ax.set_title("Top-1 Improvement vs Baselines",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x)
    # Fixed: rotation to prevent overlap
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha='right')
    if vs_fusefl:
        ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("multi_model_improvement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("      multi_model_improvement.png ✓")


def _chart_heatmap(available, all_preds, gt):
    keys   = list(available.keys())
    # Fixed: shorter labels for heatmap axes
    labels = [available[k].replace('\n', '\n') for k in keys]
    n      = len(keys)
    matrix = np.zeros((n, n))

    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i != j:
                a = hit_vector(all_preds[ki], gt, k=1)
                b = hit_vector(all_preds[kj], gt, k=1)
                matrix[i, j] = cohens_d(a, b)

    fig, ax = plt.subplots(figsize=(max(9, n * 1.3), max(7, n * 1.1)))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=0.8)
    plt.colorbar(im, ax=ax, label="Cohen's d (effect size)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=8, color='white' if v > 0.5 else 'black')

    ax.set_title("Pairwise Cohen's d at Top-1\n"
                 "(larger = bigger performance difference)",
                 fontsize=10, pad=15)
    plt.tight_layout()
    plt.savefig("multi_model_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("      multi_model_heatmap.png ✓")


if __name__ == "__main__":
    main()
