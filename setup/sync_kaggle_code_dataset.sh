#!/usr/bin/env bash
# Repackage vislm/, experiments/pillar1_patch_encoder/configs/, setup/ into the Kaggle
# Dataset that Kaggle training kernels mount (see phases/phase-2-small-train.md). Run
# this after any change to vislm/ before pushing a Kaggle training kernel.
set -euo pipefail

EXPORT_DIR="$(mktemp -d)"
trap 'rm -rf "$EXPORT_DIR"' EXIT

cp -R vislm "$EXPORT_DIR/"
cp -R experiments/pillar1_patch_encoder/configs "$EXPORT_DIR/pillar1_configs"
cp -R setup "$EXPORT_DIR/"
cp requirements.txt requirements-train.txt "$EXPORT_DIR/"
find "$EXPORT_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

cat > "$EXPORT_DIR/dataset-metadata.json" << 'EOF'
{
  "title": "vislm-research-code",
  "id": "nguyennn263/vislm-research-code",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

if kaggle datasets files nguyennn263/vislm-research-code >/dev/null 2>&1; then
  kaggle datasets version -p "$EXPORT_DIR" -m "sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" --dir-mode zip
else
  kaggle datasets create -p "$EXPORT_DIR" --dir-mode zip
fi

echo "Mounted at /kaggle/input/datasets/nguyennn263/vislm-research-code/ in kernels with dataset_sources: [\"nguyennn263/vislm-research-code\"]"
