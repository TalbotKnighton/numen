#!/usr/bin/env bash
# Build and publish Numen documentation to GitHub Pages.
#
# Usage:
#   bash scripts/publish_docs.sh            # build + deploy
#   bash scripts/publish_docs.sh --dry-run  # build only (no push)
#
# Prerequisites:
#   - git remote "origin" points to GitHub
#   - The repository has GitHub Pages enabled (Settings → Pages → Deploy from gh-pages branch)
#   - uv is installed and the docs extra is available
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

echo "==> Installing docs dependencies..."
uv pip install -e ".[docs]" --quiet

echo "==> Building documentation..."
uv run mkdocs build --strict

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "Dry-run complete. Built docs are in ./site/"
  echo "Run without --dry-run to deploy to GitHub Pages."
  exit 0
fi

echo "==> Deploying to GitHub Pages..."
uv run mkdocs gh-deploy --force

REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [[ "$REPO_URL" =~ github\.com[:/](.+?)(/|\.git)?$ ]]; then
  SLUG="${BASH_REMATCH[1]}"
  echo ""
  echo "Deployed! Docs will be live at:"
  echo "  https://${SLUG%%/*}.github.io/${SLUG##*/}/"
fi
