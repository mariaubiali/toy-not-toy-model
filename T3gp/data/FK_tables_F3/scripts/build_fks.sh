#!/usr/bin/env bash
set -euo pipefail

THEORY_ID=208

find data/yamldb -name "*.yaml" | while read -r card; do
  dataset="$(basename "$card" .yaml)"
  echo "=== $dataset ==="

  pineko theory opcards "$THEORY_ID" "$dataset"
  pineko theory ekos "$THEORY_ID" "$dataset"
  pineko theory fks "$THEORY_ID" "$dataset"
done
