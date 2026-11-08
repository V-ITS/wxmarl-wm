# EPyMARL configs for WxMARL

EPyMARL is pinned at `cbc38c09588064eab978501d0f12c2cf58fa7fc2` and used unchanged.
`wrappers/epymarl_main.py` merges configs in this order (later wins):

1. EPyMARL `src/config/default.yaml`
2. `env.yaml` (this folder; the env-level config)
3. EPyMARL `src/config/algs/<algo>.yaml`
4. `<algo>.yaml` (this folder)
5. `eval.yaml` (this folder; only with `--eval`)
6. command-line overrides (`with key=value`)

## Deviations from EPyMARL defaults

Everything not listed here is the EPyMARL default for that algorithm.

| Key | EPyMARL default | Here | Applies to | Why |
|---|---|---|---|---|
| `common_reward` | `True` | `False` | IQL, IA2C, IPPO, MAA2C, MAPPO | Individual rewards. |
| `common_reward` / `reward_scalarisation` | `True` / `sum` | unchanged, set explicitly | VDN, QMIX | Team-sum reward; EPyMARL asserts a common reward for mixers. |
| `use_rnn` | `False` | `True` | VDN, QMIX | RNN agents are required because cell motion is hidden. IQL and the actor-critic methods already default to `True`. |
| `save_model` | `False` | `True` | all | Checkpoints are needed for evaluation. |
| `save_model_interval` | 50 000 | 100 000 | all | Fewer checkpoints. |
| `use_wandb`, `wandb_team`, `wandb_project`, `wandb_mode` | `False`, null, null, `offline` | set by `scripts/train.sh` from `WANDB_MODE` (default off) and `WANDB_ENTITY`; project `wxmarl` | all | Off unless asked for. |
| `local_results_path` | `results` (relative to the working directory) | `<repo>/results` | all | Set by the launcher so output does not depend on the working directory. |

Schedule keys have no single EPyMARL default: `default.yaml` targets short SMAC debugging runs and each env config overrides them. `env.yaml` copies EPyMARL's `gymma` env config (`test_interval` 50 000, `log_interval` 50 000, `runner_log_interval` 10 000, `learner_log_interval` 10 000, `t_max` 2 050 000) except `test_nepisode`, which is 20 instead of 100. As in EPyMARL, the actor-critic algorithm configs then raise `t_max` to 20 050 000.

`eval.yaml` (evaluation only): `evaluate: True`, `runner: episode`, `batch_size_run: 1`, `save_model: False`, `env_args.render: True`, `env_args.eval_scenarios: True`. The episode runner is used so the env runs in the main process and can upload videos to wandb.

## Things EPyMARL does that affect results

- The actor-critic methods use the parallel runner with 10 env workers; the Q-learning methods use the episode runner with one env. The workers get seeds `seed + i`.
- `standardise_rewards: True` is the default for all seven algorithms. With `common_reward: False` the running statistics are kept per agent.
- `obs_agent_id: True` appends an agent one-hot to each observation (input size 48 + 12 = 60).
- There is no per-agent "done" and no agent-alive mask in the losses. After its flight ends, an agent contributes zero-observation, zero-reward samples with the forced nominal action until the episode ends (as dead agents do in SMAC).
- Episodes have variable length (about 50 steps; cap 120). The last flight ending is stored as `terminated=True`; reaching the cap is stored as a truncation.
- In-training `test_*` statistics use the same env instances as training, so they are not on the fixed evaluation scenarios. `scripts/evaluate.py` is.

## Env arguments (`env_args.*`)

| Key | Default | Meaning |
|---|---|---|
| `map_name` | `default` | Weather preset, `configs/env/<map_name>.yaml`. |
| `render` | `False` (training), `True` (`eval.yaml`) | Record the first `render.n_episodes` episodes and write replay + video. |
| `eval_scenarios` | `False` | Reset episode *k* with `eval_seeds[k]`. |
| `out_dir` | null | Output folder for replays, videos and `metrics.json`. |
| `checkpoint`, `algorithm`, `repo_commit`, `epymarl_commit` | set by the scripts | Metadata, stored in the run config and next to each replay. |
