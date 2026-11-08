#!/usr/bin/env bash
# Clone EPyMARL into third_party/epymarl at the pinned commit (see configs/epymarl/README.md).
#   bash scripts/setup_epymarl.sh
set -euo pipefail
COMMIT=cbc38c09588064eab978501d0f12c2cf58fa7fc2
cd "$(dirname "$0")/.."
DEST=third_party/epymarl

[ -d "$DEST/.git" ] || git clone https://github.com/uoe-agents/epymarl.git "$DEST"
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "EPyMARL at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
