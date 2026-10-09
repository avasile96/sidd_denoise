#!/usr/bin/env bash
# Package the deliverable: code, configs, notebooks, docs, results and figures.
# Excludes data/, weights/ and checkpoints (*.pt); training logs, summaries and the pre-fix
# results in archive_unshuffled/ are kept.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-sidd_denoise.zip}
rm -f "$OUT"
zip -qr "$OUT" \
  README.md WRITEUP.md CONCEPTS.md ACRONYMS.md pyproject.toml make_zip.sh \
  src scripts configs notebooks tests results \
  runs/*/log.csv runs/*/summary.json runs/*/config.json \
  archive_unshuffled/results archive_unshuffled/runs/*/log.csv archive_unshuffled/runs/*/summary.json archive_unshuffled/runs/*/config.json \
  -x "*/__pycache__/*" "*.pyc" "*.ipynb_checkpoints*" "*.DS_Store"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
