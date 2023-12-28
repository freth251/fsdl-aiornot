#!/bin/bash
set -uo pipefail
set +e

# tests whether we can achieve a criterion loss
# on a single batch within a certain number of epochs

FAILURE=false

# constants and CLI args set by aiming for <5 min test on commodity GPU,
#   including data download step
MAX_EPOCHS="${1:-100}"  # syntax for basic optional arguments in bash
CRITERION="${2:-1.0}"

python -m training.run_experiment \
  --batch_size 16 \
  --max_epochs "$MAX_EPOCHS"  --num_workers 2 || FAILURE=true

python -c "import json; loss = json.load(open('training/logs/wandb/latest-run/files/wandb-summary.json'))['train/loss']; assert loss < $CRITERION" || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Memorization test failed at loss criterion $CRITERION"
  exit 1
fi
echo "Memorization test passed at loss criterion $CRITERION"
exit 0
