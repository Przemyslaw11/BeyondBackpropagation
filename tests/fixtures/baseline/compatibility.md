# Existing command compatibility matrix

This is the command surface to preserve while the new CLI is introduced. The
legacy scripts remain the compatibility surface until their wrappers are retired.

| Workflow | Existing entry point | Migration status at baseline |
| --- | --- | --- |
| Single experiment | `python3 scripts/run_experiment.py --config <path>` | legacy |
| Optuna tuning | `python3 scripts/run_optuna_search.py --config <path>` | legacy |
| Local batch | `python3 scripts/run_local_array.py --config-dir <dir>` | legacy |
| SLURM single | `scripts/slurm_scripts/run_single_experiment.slurm` | legacy |
| SLURM array | `scripts/slurm_scripts/run_array.slurm` | legacy |
| SLURM Optuna | `scripts/slurm_scripts/run_optuna.slurm` | legacy |

Result and checkpoint key names are currently algorithm-specific and are kept
unchanged in the compatibility layer. New structured artifacts use explicit
units and provenance alongside those legacy return keys.

Observed legacy result vocabulary:

- Run-level: `error`, `training_duration_sec`, `total_run_duration_sec`,
  `peak_gpu_mem_used_mib`, `total_gpu_energy_joules`,
  `total_gpu_energy_wh`, `test_loss`, and `test_accuracy`.
- Profiling: `model_parameters_trainable`, `model_parameters_total`,
  `estimated_fwd_gflops`, and `estimated_bp_update_gflops`.
- CodeCarbon: `codecarbon_enabled`, `codecarbon_csv_path`,
  `codecarbon_country_iso`, `codecarbon_mode`, and
  `codecarbon_emissions_gCO2e`.
- W&B history prefixes include `BP_Baseline/*`, `FF_Hinton/*`,
  `CaFo_DFA/*`, and `Layer_W*_M*/*`; local logging uses the same names.

Checkpoint names that must remain readable by wrappers:

| Algorithm | Checkpoint pattern |
| --- | --- |
| BP | `bp_checkpoint_epoch_<n>.pth`, `bp_<experiment_name>_best.pth` |
| FF | `ff_checkpoint_epoch_<n>.pth`, `ff_<experiment_name>_best.pth` |
| CaFo | `cafo_predictor_<block_index>_complete.pth` |
| MF | `mf_matrix_M0_complete.pth`, `mf_layer_<layer_index>_complete.pth` |

Default logs are resolved as
`results/<experiment_name>/<experiment_name>_run.log` for SLURM and
`results/local/<experiment_name>/<experiment_name>_run.log` for local runs.
