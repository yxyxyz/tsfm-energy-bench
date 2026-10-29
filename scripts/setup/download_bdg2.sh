#!/usr/bin/env bash
# Download the cleaned meter and weather tables for all BDG2 sites.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/bdg2" >&2
  exit 2
fi

target=$1
repository=https://github.com/buds-lab/building-data-genome-project-2.git
include_paths="data/meters/cleaned/electricity_cleaned.csv,data/meters/cleaned/chilledwater_cleaned.csv,data/meters/cleaned/hotwater_cleaned.csv,data/weather/weather.csv"

if [[ -e "$target" && ! -d "$target" ]]; then
  echo "Target exists and is not a directory: $target" >&2
  exit 1
fi
if [[ -d "$target" ]] && find "$target" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Target directory is not empty; refusing to overwrite: $target" >&2
  exit 1
fi

command -v git >/dev/null
git lfs version >/dev/null

GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$repository" "$target"
git -C "$target" sparse-checkout init --no-cone
git -C "$target" sparse-checkout set \
  data/meters/cleaned/electricity_cleaned.csv \
  data/meters/cleaned/chilledwater_cleaned.csv \
  data/meters/cleaned/hotwater_cleaned.csv \
  data/weather/weather.csv
git -C "$target" checkout master
git -C "$target" lfs pull --include="$include_paths"

for relative_path in \
  data/meters/cleaned/electricity_cleaned.csv \
  data/meters/cleaned/chilledwater_cleaned.csv \
  data/meters/cleaned/hotwater_cleaned.csv \
  data/weather/weather.csv; do
  file="$target/$relative_path"
  if [[ ! -s "$file" ]]; then
    echo "Missing or empty downloaded file: $file" >&2
    exit 1
  fi
  if head -n 1 "$file" | grep -q 'git-lfs.github.com/spec'; then
    echo "Git LFS pointer remains instead of real data: $file" >&2
    exit 1
  fi
done

echo "BDG2 meter and weather files downloaded successfully under: $target/data"
