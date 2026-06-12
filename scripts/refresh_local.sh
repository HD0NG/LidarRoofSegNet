#!/usr/bin/env bash
# Local post-pull refresh (Mac, no PyG). Run after pulling the deterministic
# eval batch (reproduce.config.js) from the server. Recomputes everything that
# is derived locally from the pulled result JSONs and re-renders the figures,
# then prints a headline-number digest so any shift vs the manuscript is easy
# to catch.
#
# Pure NumPy/scipy/lightgbm/matplotlib — nothing here touches PyG or the model,
# so it runs on the Mac. Server-side outputs (table JSONs, .npz, B3D audit) must
# already be in place; this only refreshes the locally-computed derivatives.
set -e
PYTHON="${PYTHON:-/opt/miniconda3/envs/sb310/bin/python}"
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root

echo "=== [1/4] paired significance tests (-> significance_tests.json) ==="
"$PYTHON" scripts/analyze_significance.py >/dev/null
echo "    done"

echo "=== [2/4] feature-subset ablation (-> feature_ablation.json) ==="
"$PYTHON" scripts/analyze_feature_ablation.py >/dev/null
echo "    done"

echo "=== [3/4] re-render figures (-> paper/figures/*.pdf,*.png) ==="
"$PYTHON" scripts/plot_feature_importance.py >/dev/null
"$PYTHON" scripts/plot_qualitative.py >/dev/null
echo "    done"

echo "=== [4/4] headline-number digest (verify against paper/FINISH_PROMPT.md) ==="
"$PYTHON" - <<'PY'
import json
from pathlib import Path

R = Path("evaluation_results")

def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None

def row(label, s, keys=("mean_number_of_faces_error", "mean_ARI", "mean_mPrec")):
    if s is None:
        print(f"  {label:<26} (missing)")
        return
    vals = "  ".join(f"{k.split('_')[-1] if k!='mean_number_of_faces_error' else 'NFE'}={s[k]:.3f}" for k in keys)
    print(f"  {label:<26} {vals}")

print("\n roofNTNU main ablation (test):")
row("raw", load("ablation_raw_summary.json"))
row("hand-tuned", load("ablation_handtuned_summary.json"))
row("LightGBM @ 0.6", load("sweep_thresh_060_summary.json"))

sig = load("significance_tests.json")
if sig:
    hl = {r["metric"]: r for r in sig["handtuned_vs_lgb06"]}
    nfe, ari = hl.get("number_of_faces_error"), hl.get("ARI")
    print("\n significance hand-tuned -> LightGBM (expect NOT significant):")
    if nfe: print(f"  NFE  p={nfe['wilcoxon_p']:.3f}  (W/L/T {nfe['n_better']}/{nfe['n_worse']}/{nfe['n_tie']})")
    if ari: print(f"  ARI  p={ari['wilcoxon_p']:.3f}")

# Val operating-point selection (argmax ARI on val).
print("\n val threshold selection (argmax ARI -> operating point):")
best_t, best_ari = None, -1
for tag, t in [("050","0.5"),("060","0.6"),("070","0.7"),("080","0.8")]:
    s = load(f"sweep_val_thresh_{tag}_summary.json")
    if s is None: continue
    a, n = s["mean_ARI"], s["mean_number_of_faces_error"]
    mark = ""
    if a > best_ari: best_ari, best_t = a, t
    print(f"  tau={t}: ARI={a:.3f} NFE={n:.3f}")
if best_t: print(f"  -> selected tau={best_t} (should be 0.6)")

print("\n Building3D external validation (n=514, transfer):")
row("raw", load("building3d_raw_summary.json"))
row("hand-tuned", load("building3d_handtuned_summary.json"))
row("LightGBM @ 0.6", load("building3d_lgb06_summary.json"))

fa = Path("artifacts/merge_classifier/feature_ablation.json")
if fa.exists():
    a = json.load(open(fa)); full = a["full_reference"]["pr_auc"]
    print("\n feature-subset ablation (val PR-AUC, % of full):")
    for r in a["results"]:
        print(f"  top-{r['k']:>2}: PR-AUC={r['val_metrics']['pr_auc']:.3f} ({100*r['val_metrics']['pr_auc']/full:.0f}%)")
PY

echo ""
echo "Refresh complete. If any digit above differs from paper/FINISH_PROMPT.md,"
echo "update the inline numbers there before running the manuscript session."
exit 0
