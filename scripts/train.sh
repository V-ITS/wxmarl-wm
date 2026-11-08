#!/usr/bin/env bash
# Launch EPyMARL training on WxMARL.
#   bash scripts/train.sh <algo> <seed> <preset> [sacred overrides, e.g. t_max=200000]
# algo: iql | vdn | qmix | ia2c | ippo | maa2c | mappo
# wandb is off by default; WANDB_MODE=online or offline turns it on (run name: <algo>_<preset>_seed<seed>[_$RUN_TAG]; team: WANDB_ENTITY).
set -euo pipefail
ALGO=${1:?usage: train.sh <algo> <seed> <preset> [overrides]}
SEED=${2:?seed}
PRESET=${3:-default}
shift $(( $# < 3 ? $# : 3 ))
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

case "${WANDB_MODE:-disabled}" in
  online | offline) WANDB_ARGS=(use_wandb=True "wandb_mode=$WANDB_MODE" "wandb_team=${WANDB_ENTITY:-null}") ;;
  disabled) WANDB_ARGS=(use_wandb=False) ;;
  *) echo "WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;;
esac
export WANDB_TAGS="$ALGO,seed$SEED,$PRESET"
# Run name shown in wandb; RUN_TAG (optional) labels the variant, e.g. RUN_TAG=grid32-1nm.
export WANDB_NAME="${WANDB_NAME:-${ALGO}_${PRESET}_seed${SEED}${RUN_TAG:+_$RUN_TAG}}"

exec python -m wrappers.epymarl_main --config="$ALGO" with \
  seed="$SEED" env_args.map_name="$PRESET" "${WANDB_ARGS[@]}" "$@"
