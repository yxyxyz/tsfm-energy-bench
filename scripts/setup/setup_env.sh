#!/usr/bin/env bash
# Create or update one conda environment from a requirements file.
#
# Run this on the GPU host after syncing the repository:
#
#   bash scripts/setup/setup_env.sh cikm-timesfm30 requirements/timesfm30.txt
#   bash scripts/setup/setup_env.sh cikm-tsfm-modern requirements/modern.txt
#   bash scripts/setup/setup_env.sh cikm-timesfm10 requirements/timesfm10.txt
#
# Keep the paper's pinned 2.5 dependencies separate from the 3.0 extension.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 ENV_NAME REQUIREMENTS_FILE" >&2
  exit 2
fi

env_name=$1
requirements=$2
project_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$project_dir"

if [[ ! -f "$project_dir/$requirements" ]]; then
  echo "Requirements file not found: $project_dir/$requirements" >&2
  exit 2
fi

if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
  echo "Environment $env_name already exists; installing requirements into it."
else
  conda create -y -n "$env_name" python=3.11
fi

conda run -n "$env_name" python -m pip install --upgrade pip
conda run -n "$env_name" python -m pip install -r "$project_dir/$requirements"

echo "Environment $env_name is ready."
