#!/usr/bin/env bash
# Run inside the project's Linux development container.
set -euo pipefail
replay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$replay_root"
export TPG="$replay_root"
cmake --build build --target TPGExperimentMPI --parallel 2
python3 scripts/replay_test_table.py prepare --environment Acrobot
python3 scripts/replay_test_table.py run --environment Acrobot --episodes 20
python3 scripts/plot/plot_test_boxplots.py --environment Acrobot
