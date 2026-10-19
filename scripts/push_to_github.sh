#!/usr/bin/env bash
# Run manually from the repository root after reviewing the files.
set -euo pipefail
git init
git branch -M main
git add .
git commit -m "Implement budgeted DRL context acquisition for POI recommendation"
gh auth status
gh repo create sepehrmustafavi/drl-context-poi --public --source=. --remote=origin --push
