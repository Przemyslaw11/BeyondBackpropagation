# Implementation prompts — one per phase (PPAM 2026 camera-ready)

Each prompt is SELF-CONTAINED. The receiving agent has none of the planning conversation.
Shared context block is repeated deliberately.

---
## SHARED CONTEXT (paste at the top of every prompt)

Repo: /Users/pspyra/Desktop/UNI/BeyondBackpropagation (Przemyslaw11/BeyondBackpropagation, branch main)
Paper: "Energy-Efficient Deep Learning Without Backpropagation" — ACCEPTED at PPAM 2026,
Springer LNCS. Producing the camera-ready (15 pages max) + response-to-reviewers letter.
Deadline 2 November 2026.
The study benchmarks three forward-only algorithms — Forward-Forward (FF), Cascaded
Forward (CaFo, Rand-CE and DFA-CE variants), Mono-Forward (MF) — against tuned
backpropagation (BP) baselines on MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100, measuring
accuracy, wall time, GPU energy and peak GPU memory.
Headline claims: MF matches/surpasses BP accuracy on MLPs, with up to 41% lower energy and
34% faster training on the CIFAR-10 3x2000 MLP.

Key paths:
- Manuscript: latex_source/PPAM_2026_SUBMISSION/main.tex (+ bibliography.bib, plots/)
- Reviews: latex_source/review.tex
- Master thesis (rich source of ready text): latex_source/original_master_thesis/
  main.tex holds Appendix A (hyperparameter tuning) and Appendix B (environment);
  tex/4_ExperimentsAndResults.tex holds the full results tables.
- Code: src/{algorithms,architectures,baselines,data_utils,training,tuning,utils}
- Configs: configs/{base.yaml,bp_baselines,cafo,ff,mf,tuning}
- Scripts: scripts/{run_experiment.py,run_optuna_search.py,run_local_array.py,slurm_scripts}

Cluster: PLGrid / ACK Cyfronet AGH "Athena", SLURM, NVIDIA A100-SXM4-40GB.
SLURM directives in use: -A plglscclass26-gpu-a100, -p plgrid-gpu-a100, --gres=gpu:1,
--cpus-per-task=16, --mem=128000M, --time=08:00:00.
NOTE: the grant changed. `plgoncotherapy-gpu-a100` no longer exists; the only funded grant
is `plglscclass26` (valid 2026-03-02 -> 2027-03-01). Any older prompt text naming
`plgoncotherapy-gpu-a100` is stale.

Environment — the thesis records ONE environment, but the cluster has since moved. Both
matter, and they are NOT the same:

| Component    | Thesis Appendix B (published results) | Measured on Athena, Phase 0 |
|--------------|---------------------------------------|-----------------------------|
| OS           | Rocky Linux 9.5                       | **Rocky Linux 9.8**         |
| NVIDIA driver| 570.86.15                             | **595.71.05**               |
| CPU          | AMD EPYC 7742                         | AMD EPYC 7742 (same)        |
| Python       | 3.10.4                                | 3.10.4 (same)               |
| PyTorch      | 2.4.0+cu121                           | 2.4.0+cu121 (same)          |
| torchvision  | 0.19.0                                | 0.19.0+cu121 (same)         |
| CUDA toolkit | 12.4.0                                | 12.4.0 (same)               |
| optuna       | 4.2.1                                 | **4.9.0**                   |
| wandb        | 0.19.8                                | **0.29.0**                  |
| pynvml       | 12.0.0                                | **13.0.1** (+ nvidia-ml-py 13.610.43) |
| codecarbon   | 3.0.1                                 | **3.3.0**                   |
| numpy        | 2.2.4                                 | **2.2.6**                   |
| scikit-learn | 1.6.1                                 | **1.7.2**                   |

Shell convention: prefix every shell command with `rtk` (token-filtering proxy),
e.g. `rtk git status`, `rtk pytest tests/`.

House rules:
- Comments only where the code cannot speak for itself; one short line. No essays.
- Do not create markdown summary files.
- Do not refactor, rename or "improve" anything outside the stated scope.
- Do not commit or push unless explicitly asked.

---
## PHASE 0 — Environment validation and go/no-go gate  [COMPLETE — verdict: **GO**]

> **STATUS: EXECUTED 2026-09-05. Do not re-run.** Retained for provenance. The findings
> below are inputs to later phases; where they contradict a later prompt, THE FINDINGS WIN.
>
> **Verdict: GO.** All four STOP conditions cleared.
> - Grant `plglscclass26-gpu-a100`: **3616.22 GPU-h remaining** vs ~155 needed (~23x headroom).
> - Partition `plgrid-gpu-a100` UP, but **13 of 48 nodes in `fail*`, 0 idle** — capacity, not
>   entitlement, is the binding constraint on the calendar.
> - **MIG disabled**; all NVML power/memory calls work under pynvml 13.0.1.
> - Smoke job completed end to end, exit 0, artefacts + W&B all present.
>
> **Findings that change later phases:**
> - **Driver is now 595.71.05, not 570.86.15.** Affects Phase 1 task 4d and Phase 4.
> - **All four NVML calls Phase 2 needs are SUPPORTED** (`GetUtilizationRates`,
>   `GetTemperature`, `GetClockInfo`, `GetComputeRunningProcesses`). Phase 2 task 11 is safe.
> - **Two bonus instruments confirmed available:** `nvmlDeviceGetTotalEnergyConsumption`
>   (hardware joule counter, no integration error) and per-process GPU bytes from
>   `GetComputeRunningProcesses`.
> - **Compute nodes HAVE outbound internet** (`api.wandb.ai:443` reachable). W&B runs online;
>   the offline/sync path is therefore not on the critical path.
> - **E3 CONFIRMED: instrument misattribution is real.** All four of the paper's hardware
>   traces are W&B-agent `system.*` keys, not NVML. "Process Memory In Use" is
>   `system.proc.memory.rssMB` — **host RSS, not GPU memory**.
> - **W&B system metrics do not start until ~30 s into a run** (~7.5 s sampling period), so
>   short HPO trials get NO hardware trace. Use the project's own NVML instrumentation.
> - **`peak_gpu_mem_used_mib` is a weak metric** — it is device-wide and dominated by the
>   ~800 MiB CUDA context (BP 923.8 vs MF 905.8 MiB). It cannot support a memory-efficiency
>   claim at this scale. Phase 2 should switch to `torch.cuda.max_memory_allocated()`.
> - **`$HOME` is only 10 GiB and a single BP run filled it** via per-epoch checkpoints.
>   Fixed by a `keep_best_only` policy (7.5 GB -> ~57 MB). Checkpoints now live on `$SCRATCH`.
> - **`$SCRATCH` auto-purges at 30 days** and currently hosts the 5.5 GB venv — it is due for
>   deletion mid-campaign. **Unresolved risk.**
> - **`plgrid-now` cannot run arrays** (`MaxSubmitJobsPU=1`) but carries a 15x priority
>   weight — use it for probes and single short runs only. The 480-run array must go to
>   `plgrid-gpu-a100`.
> - **Never resubmit a pending job to jump the queue** — AGE is the only priority factor that
>   accrues here (FAIRSHARE is ~4). Use `scontrol update` to shrink a job in place.
>
> **Deviations from this prompt's constraints, authorised by the user mid-phase:** the
> "write no project code" rule was overridden to apply the `keep_best_only` fix
> (`src/utils/helpers.py`, `src/baselines/bp.py`, `src/algorithms/ff.py`, `configs/base.yaml`)
> and to correct `WANDB_MODE`/grant in `scripts/slurm_scripts/*.slurm`. All changes are
> uncommitted working-tree edits.
>
> **TOP PRE-PHASE-2 ACTION: commit the working tree.** Committed `main` currently cannot
> submit a job (dead grant) and cannot run FF at all (an evaluation-kwargs bug is fixed only
> in the working tree).

<SHARED CONTEXT>

### Why this phase exists
Every later phase assumes the Athena environment still works exactly as it did when the
original experiments ran. That assumption is untested and several parts of it are known to
decay: PLGrid grants expire, GPU drivers get upgraded, compute nodes may have no outbound
internet, and SLURM array/QOS limits decide whether a 480-run campaign takes three days or
three weeks. The whole ~111 GPU-h budget and the 8-week timeline rest on runtime figures
taken from the accepted paper, not from a measurement made today.

This phase writes NO project code and runs NO real experiments. It produces a single
go/no-go report. If anything in group A or C fails, STOP and report — do not work around it.

### Preconditions
None. This runs before everything, including Phase 1.

### Read first
README.md; scripts/slurm_scripts/ (all files); scripts/run_experiment.py;
requirements.txt; src/utils/monitoring.py; configs/base.yaml;
latex_source/original_master_thesis/main.tex Appendix B (the recorded environment).

### Discover, do not assume
The login hostname, module names, virtualenv/conda path and the location of the repository
clone on the cluster are NOT recorded in this prompt. Derive them from README.md and the
SLURM scripts. If they are not discoverable, ASK THE USER rather than guessing. Never
invent a hostname.

### Secret handling — mandatory
Never read, cat, grep or print .env files or any credential file. Load secrets into shell
variables and print only redacted or derived results (e.g. "WANDB_API_KEY present, length
40, prefix ab…"). Never echo a key, token or password into the transcript or into a log.

---
### GROUP A — Access and entitlement (HARD GATE)
A1. Establish SSH access to the Athena login node. Confirm non-interactive access works
    (key-based, no password prompt), since later phases will submit jobs unattended.
A2. VERIFY THE GRANT IS STILL VALID AND FUNDED. The SLURM scripts use
    `-A plgoncotherapy-gpu-a100`. PLGrid grants have end dates and GPU-hour allocations
    that expire. Use the PLGrid grant-inspection tooling available on the cluster
    (e.g. `hpc-grants`, or `sacctmgr show assoc` filtered to the user) to report: grant
    name, validity window, GPU-hours allocated, GPU-hours consumed, GPU-hours REMAINING.
    THIS IS THE SINGLE MOST IMPORTANT NUMBER IN THIS PHASE. The plan needs ~111 GPU-h,
    ~155 with contingency. If remaining hours are below ~155, STOP and report immediately —
    the whole plan must be re-scoped.
A3. Confirm the partition `plgrid-gpu-a100` exists and is accepting jobs (`sinfo` for that
    partition: node count, state, and any drain/down nodes).
A4. Report the QOS and limits that apply to this account: MaxSubmitJobs, MaxJobsPerUser,
    max concurrent GPU jobs, and SLURM's `MaxArraySize`. Phase 3 submits a 480-element
    array (4 configs x 6 rungs x 20 seeds); Phase 4 adds more. If MaxArraySize is below
    what the campaign needs, report the chunking that will be required.
A5. Report the maximum wall clock permitted on this partition. The existing scripts request
    `--time=08:00:00`; confirm that is allowed and whether longer is possible.

### GROUP B — Software environment
B1. Locate the Python environment used by the original experiments. Report Python version,
    and the installed versions of: torch, torchvision, numpy, optuna, wandb, pynvml,
    codecarbon, scikit-learn, PyYAML, tqdm.
B2. DIFF against the environment recorded in thesis Appendix B: Python 3.10.4,
    PyTorch 2.4.0+cu121, torchvision 0.19.0, numpy 2.2.4, optuna 4.2.1, wandb 0.19.8,
    pynvml 12.0.0, codecarbon 3.0.1, scikit-learn 1.6.1, PyYAML 6.0.2, tqdm 4.67.1,
    CUDA Toolkit 12.4.0. Report EVERY discrepancy. Do not upgrade or downgrade anything
    yet — just report.
B3. Confirm `import torch; torch.cuda.is_available()` is True on a COMPUTE node (not the
    login node) and report `torch.version.cuda` and the detected device name.
B4. Report the state of the repository clone on the cluster: path, branch, commit hash, and
    whether it is dirty. Compare the commit to local `main`.
B5. Report free space and quota on $SCRATCH, $HOME and any group storage. Phase 3 alone
    produces 480 runs of per-run CSVs plus checkpoints; estimate whether headroom is
    sufficient and say so explicitly.

### GROUP C — Hardware and instrumentation (HARD GATE)
C1. On a compute node, run `nvidia-smi` and report: GPU model, memory, and the DRIVER
    VERSION. The recorded driver is 570.86.15. IF THE DRIVER HAS CHANGED, FLAG IT
    PROMINENTLY — driver version governs NVML power-reading semantics, so a change means
    new energy measurements may not be directly comparable to the published ones, and the
    Phase 4 reproduction check becomes the deciding evidence.
C2. Report whether MIG (Multi-Instance GPU) is enabled. If it is, NVML power queries behave
    differently or fail, and the energy methodology is affected. This is a hard gate.
C3. Verify the NVML calls the project depends on actually return sane values on a compute
    node: `nvmlDeviceGetHandleByIndex`, `nvmlDeviceGetPowerUsage`, `nvmlDeviceGetMemoryInfo`.
    Report a few sampled values with units.
C4. Verify the NVML calls Phase 2 will ADD also work, since the entire figure rebuild
    depends on them: `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetTemperature`,
    `nvmlDeviceGetClockInfo`, `nvmlDeviceGetComputeRunningProcesses`. If any raises
    NVMLError_NotSupported or NVMLError_NoPermission, report exactly which — Phase 2 and
    Phase 5 must be re-planned around it.
C5. Verify `psutil` process RSS reads correctly on a compute node. Phase 3's host-side
    activation cache is unreportable without it.
C6. Confirm the GPU is allocated exclusively under `--gres=gpu:1`, i.e. that no other
    process appears in `nvmlDeviceGetComputeRunningProcesses` during a job. Device-level
    power readings are only attributable to our workload if the GPU is not shared.

### GROUP D — Data and network reachability
D1. Determine whether COMPUTE nodes have outbound internet access. Test explicitly; do not
    infer from the login node, which usually does have it. This single fact determines
    whether W&B must run offline and whether datasets must be pre-staged.
D2. Confirm MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 are already present in the
    dataset directory the configs expect. If any are missing and compute nodes lack
    internet, they must be downloaded from the login node first — report this as a required
    pre-step for Phase 2.
D3. Verify the dataset directory is readable from a compute node and report its path and
    total size.

### GROUP E — Weights & Biases
E1. Confirm a W&B credential is available in the environment (redacted reporting only, per
    the secret-handling rule above).
E2. Confirm the project `przspyra11/BeyondBackpropagation` is reachable and report: number
    of runs, date of the earliest and most recent run, and whether full run HISTORIES (not
    just summaries) are still retained. Phase 4 task 4 depends on these histories for the
    reproduction cross-check and for confirming the instrument misattribution.
E3. Report which metric keys exist in a representative historical run. Specifically look
    for the W&B SYSTEM panels the current paper figures were exported from — GPU
    utilisation, GPU temperature, SM clock, and "Process Memory In Use" — and confirm
    whether these are `system/*` keys logged by the W&B agent rather than metrics logged
    by our own code. This confirms or refutes the finding that three of the paper's four
    hardware traces did not come from NVML.
E4. If compute nodes lack internet (D1), verify that `WANDB_MODE=offline` works during a
    job and that `wandb sync` from the login node afterwards successfully uploads the run.
    Test this end to end — an untested offline/sync path is a campaign-scale risk.

### GROUP F — End-to-end smoke test
F1. Submit ONE real SLURM job using the existing scripts and an existing config, reduced to
    a single epoch or a small step cap. Use a cheap config (an MNIST MLP).
F2. Confirm the job: is accepted by the scheduler; starts; writes stdout/stderr into
    slurm_logs/ where the scripts expect; completes with exit code 0; produces the usual
    result artefacts; and appears in W&B (directly, or after sync).
F3. Confirm the NVML energy monitor produced a non-zero, plausible energy figure and that
    the background sampling thread shut down cleanly. Report the sampled interval actually
    achieved versus the configured `monitoring.energy_interval_sec` of 0.2 s.
F4. Test the `EXPERIMENT_SEED` environment-variable override (src/training/engine.py ~L177)
    by submitting the same config twice with two different seeds and confirming the logs
    show the override taking effect and the results differ. Phase 3's entire 2-D array
    design depends on this working.
F5. Submit a SMALL SLURM ARRAY (e.g. 4 elements over 2 configs x 2 seeds) to validate the
    2-D indexing pattern Phase 2 will generalise. Note: the current
    scripts/slurm_scripts/run_array.slurm is known-broken — it declares `--array=1-25`
    against 27 config entries and references a nonexistent
    configs/ff/mnist_mlp_3x1000_SGD_ref.yaml. Do NOT fix it here; Phase 2 owns that. Just
    confirm the array mechanism itself works.

### GROUP G — Budget calibration (determines whether the plan is feasible)
G1. Time ONE full run (not one epoch) of each of: a BP MLP, an MF MLP, an FF MLP, and a
    CaFo CNN, using existing configs and existing hyperparameters. Report actual wall-clock
    seconds for each.
G2. Compare against the figures the compute budget was built on: BP CIFAR-10 3x2000
    ~268 s, MF CIFAR-10 3x2000 ~178 s, FF MNIST 4x2000 ~575 s, BP MNIST 4x2000 ~43 s.
G3. RECOMPUTE THE CAMPAIGN BUDGET from your measured times and report whether ~111 GPU-h
    (~155 with contingency) still holds. Break it down as: Phase 3 HPO (16 studies x 50
    trials at a reduced epoch budget), Phase 3 finals (480 runs), Phase 4 HPO, Phase 4
    finals. If measured times are materially higher, say by how much and which phase
    breaks first.
G4. Using the QOS limits from A4, estimate the WALL-CLOCK duration of the campaign, not
    just the GPU-hours. If only N jobs may run concurrently, the 8-week timeline may not
    survive even when the GPU-hour budget does. This is the number the schedule actually
    depends on.

---
### Constraints
- Write NO project code. Create no files under src/, configs/ or latex_source/.
  Throwaway diagnostic scripts are fine; put them in a scratch location and say where.
- Do NOT fix any bug you find. Report it. Phase 2 owns the fixes.
- Do NOT upgrade, downgrade or install packages without asking. Report drift only.
- Do NOT delete anything, including old runs, logs or checkpoints — they may be the only
  surviving record of the published results.
- Do NOT submit long or expensive jobs. Total Phase 0 consumption should stay under
  roughly 1 GPU-h, excluding the G1 timing runs.

### Acceptance criteria
A single go/no-go report containing:
- A pass/fail line for every check A1-G4.
- GPU-hours REMAINING on the grant versus the ~155 required.
- Driver version now versus 570.86.15, and whether MIG is on.
- Which of the four new NVML calls Phase 2 depends on are actually supported.
- Whether compute nodes have internet, and whether the W&B offline/sync path was proven
  to work end to end.
- Measured per-run wall times and a recomputed campaign budget in both GPU-hours and
  calendar wall-clock.
- An explicit GO or NO-GO recommendation with the blocking items named.

### STOP conditions — report immediately, do not work around
- Grant expired, or remaining GPU-hours below ~155 (A2).
- Partition unavailable or the account cannot submit to it (A3).
- MIG enabled, or NVML power queries unsupported/unpermitted (C2, C3).
- The smoke job cannot complete end to end (F2).

### Report back
The go/no-go report above, plus a prioritised list of everything that must change before
Phase 2 can start, and any assumption in the plan that your measurements contradict.

---
## PHASE 1 — Zero-compute manuscript fixes and safety net

<SHARED CONTEXT>

### Preconditions
None in terms of other phases. This runs in parallel with Phase 2 and touches ONLY LaTeX.
Phase 0 is complete (verdict GO); its measured environment is in the shared context above
and **overrides** any environment figure quoted elsewhere in this prompt.

### Toolchain — ALREADY INSTALLED AND VERIFIED (2026-09-05)
TeX Live 2026 is installed via the Homebrew **formula** (`brew install texlive`) — no sudo,
no cask. Confirmed present:
  pdflatex  pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026/Homebrew)  at /opt/homebrew/bin
  bibtex    BibTeX 0.99e
  latexmk   Version 4.87
A full `pdflatex -> bibtex -> pdflatex -> pdflatex` cycle on the UNMODIFIED paper already
succeeds: rc=0 on every pass, **0 overfull boxes**, output **14 pages**.
**BASELINE IS 14 PAGES AGAINST A 15-PAGE LIMIT — you have ONE page of headroom** for
everything tasks 4 and 5 add (environment prose, NVML definition, architecture details and
a whole new results table). This will be tight. If you exceed 15, STOP and report; do not
silently cut content to fit.
`llncs.cls` and `splncs04.bst` are vendored in the paper directory, so the LNCS class is
not a dependency you need to fetch.
Ignore the pip package named `pdflatex` — it is only a wrapper around a real binary and
builds nothing on its own.

### Read first
latex_source/PPAM_2026_SUBMISSION/main.tex (whole file, ~850 lines),
latex_source/PPAM_2026_SUBMISSION/bibliography.bib,
latex_source/original_master_thesis/main.tex — Appendix A, and Appendix B at ~lines 705-730,
latex_source/original_master_thesis/tex/4_ExperimentsAndResults.tex (lines 100-230),
latex_source/review.tex.

### Verified anchors (confirmed present — use these, do not hunt)
- `fig:ff_resource_utilization` — label at main.tex:509, referenced at :473
- `fig:cafo_convergence` — label at main.tex:550, referenced at :528
- `fig:mf_hardware_and_memory` — label at main.tex:652, referenced at :624 and :628
- `\cite{lee2015deeply}` at main.tex:822 — one of **FIFTEEN** undefined keys (see task 6)
- `\bibliography{bibliography}` at main.tex:930, `\bibliographystyle{splncs04}` at :928
- Existing environment prose at main.tex:378-380 (PyTorch/CUDA only) — this is what task 4d extends
- Thesis driver version at original_master_thesis/main.tex:713

### Tasks
1. FIX THE FIGURE 3 LABEL SWAP. In the `fig:cafo_convergence` float (main.tex:536-541).
   Filename capitalisation was ALREADY NORMALISED before handoff: the files are now
   `plots/CaFo_1.png` and `plots/CaFo_2.png`, and both `\includegraphics` paths were
   corrected to match. (Previously main.tex requested `CaFo_2.png` and `Cafo_1.png` while
   disk held `Cafo_2.png` and `CaFo_1.png` — invisible on macOS's case-insensitive
   filesystem, but it would have failed to resolve either image on Springer's or arXiv's
   case-sensitive Linux build. Do not reintroduce it.)
   THE CAPTION SWAP IS STILL OUTSTANDING. Current state:
     main.tex:538   `\subfloat[Validation Accuracy]` -> `plots/CaFo_2.png`
     main.tex:540   `\subfloat[Validation Loss]`     -> `plots/CaFo_1.png`
   Open BOTH images, read the titles rendered inside them, and make each `\subfloat[...]`
   caption match the image it actually wraps. Trust the pixels, not this prompt. This is a
   factual error in the accepted PDF.
2. DELETE the sentence in the `fig:ff_resource_utilization` caption (main.tex:509) asserting
   "Trace labels follow the `WiMj` convention: `W`i = weight matrix i, `M`j = monitor
   channel j". The legend contains only BP and FF; the sentence describes nothing.
3. DELETE the claim in the `fig:mf_hardware_and_memory` panel-(c) caption (main.tex:652)
   that memory shows a "step-like pattern ... allocations increase in discrete steps".
   MF_5.png shows a monotonically decreasing trace with no steps. Replace with a factual
   description of what the image actually shows.
4. BACK-PORT from the thesis, adapting to LNCS style and the paper's macros:
   a. NVML definition on first use (closes reviewer R3a). Thesis abstract has it verbatim:
      "direct measurements at the hardware level obtained via the NVIDIA Management
      Library (NVML) API". Thesis Appendix B adds "via the `pynvml` package". State what
      NVML does and why it was chosen.
   b. A one-sentence formal definition of "hardware-validated comparison" (closes R4a).
      Thesis supplies the substance.
   c. Architecture and optimiser details from Appendix A into Section 3 / Section 4
      (closes R2b): CaFo predictions aggregated by summation; FF goodness threshold
      theta = 2.0, peer-normalisation factor 0.03, momentum 0.9; MF weight decay fixed at
      0.0; Kaiming uniform initialisation throughout; Adam for CaFo predictors and MF
      layers versus AdamW for BP.
   d. Computational environment from Appendix B into Section 4, extending main.tex:378-380.
      **USE THE CURRENT DRIVER, 595.71.05 — NOT the thesis's 570.86.15.** (User decision,
      2026-09-05.) By camera-ready every reported number will have been regenerated by
      Phases 3 and 4 on the present cluster, so the environment section must describe THAT
      environment: Rocky Linux 9.8, NVIDIA driver 595.71.05, CUDA toolkit 12.4.0,
      PyTorch 2.4.0+cu121 (CUDA 12.1 runtime), Python 3.10.4, AMD EPYC 7742.
      Driver version must appear because it governs NVML power-reading semantics.
      ONE CAVEAT to keep straight: any table still carrying pre-regeneration numbers is
      describing 570.86.15 while this section says 595.71.05. That applies to the CaFo
      safety-net table you add in task 5 — its table note must mark the numbers provisional
      pending Phase 4, which task 5 already requires. Do NOT put two driver versions in the
      paper; one environment section, current hardware, plus a provisional-data note.
5. ADD A CaFo RESULTS TABLE to Section 5.2 (closes R3b). Source the numbers from
   latex_source/original_master_thesis/tex/4_ExperimentsAndResults.tex lines 125-210.
   Columns must match the existing `tab:ff_bp_mlp_summary` and
   `tab:mf_bp_perf_eff_summary`: Accuracy (%), Time (s), Energy (Wh), Peak Memory (MiB),
   all as mean +/- std. Mark clearly in a table note that these are 3-seed numbers pending
   the Phase 4 re-runs.
   Sanity anchors from the thesis: MNIST accuracies 98.62+/-0.02, 99.02+/-0.05,
   98.94+/-0.15; CIFAR-10 energy 5.50 / 27.37 / 6.81 Wh.
   These anchors were INDEPENDENTLY RECONFIRMED in Phase 0 by reparsing the 81 archived
   run logs (27 configs x 3 seeds), so they are trustworthy. If your table disagrees with
   them, your table is wrong.
   If you add a Peak Memory column, add a one-line footnote that it is a device-wide NVML
   reading and therefore includes the CUDA context — Phase 0 showed the metric barely
   separates algorithms (BP 923.8 vs MF 905.8 MiB). Do not let the table imply a stronger
   memory claim than the instrument supports.
6. FIX THE BIBLIOGRAPHY — **THIS IS FAR BIGGER THAN THE ONE ENTRY EARLIER DRAFTS CLAIMED.**
   A real build (TeX Live 2026, run 2026-09-05) proves `bibliography.bib` defines only
   **18 of the 30 keys** main.tex cites. **15 keys are undefined**, and the ACCEPTED
   `main.pdf` consequently renders **22 `[?]` markers**. This is a live defect in the
   accepted paper and it MUST be fixed for camera-ready.
   The 15 missing keys:
     devlin2019bert   ishikawa2025local   jouppi2017datacenter   journe2023hebbian
     lee2015deeply    lorberbom2024layer  malladi2023fine        patterson2021carbon
     papachristodoulou2024convolutional   pinchetti2022predictive
     ren2023scaling   salvatori2022associative   salvatori2022brain
     salvatori2024stable   strubell2019energy
   Recovery routes, cheapest first:
   a. `patterson2021carbon` and `strubell2019energy` exist verbatim in
      latex_source/original_master_thesis/bibliography.bib — copy them across.
   b. `devlin2019bert` is cited here, but the thesis bib defines `devlin2018bert`. Same
      paper (BERT: arXiv 2018, NAACL 2019). Copy it and reconcile key and year to the venue
      you actually cite; do not leave two variants in the file.
   c. The remaining 12 must be written from scratch.
   **DO NOT INVENT CITATIONS.** Verify each against the real publication — authors, venue,
   year, DOI where available. A fabricated reference in a Springer camera-ready is far worse
   than a missing one. If you cannot verify an entry, leave it undefined and report it by
   name rather than guessing.
   `lee2015deeply` should be Lee et al., "Deeply-Supervised Nets", AISTATS 2015 — confirm
   before entering it.
   Definition of done: the build reports ZERO undefined citations and the rendered PDF
   contains ZERO `[?]` markers (check with `pdftotext main.pdf - | grep -c '\[?\]'`).
7. REPORT (do not fix) the T11 discrepancy: paper Table 2 gives CIFAR-10 MF accuracy
   SD = 0.28 while the thesis gives 0.05 for the same cell. Phase 3 resolves this with
   fresh data. Just flag it in your final message.

### Constraints
- LNCS style via llncs.cls. UK English throughout.
- Use the paper's existing macros: \FF{}, \MF{}, \BP{}, \CaFo{}, \CaFoDFA{}.
- Units via siunitx. Tables via booktabs. Match existing table formatting exactly.
- Do NOT renumber, reorder or remove any existing float.
- Do NOT touch any file under src/, configs/ or scripts/. Those hold uncommitted Phase 0
  fixes; a stray edit there risks losing work that exists nowhere else.
- Do NOT regenerate any figure image — that is Phase 5.
- Do NOT commit or push unless explicitly asked.

### Acceptance criteria
- A full `pdflatex` -> `bibtex` -> `pdflatex` x2 cycle completes with zero undefined
  references and zero undefined citations, and `pdftotext main.pdf - | grep -c '\[?\]'`
  returns 0 (it is currently 22).
- Figure 3's subfloat captions match the content of the images, verified by opening them.
- A CaFo table exists in Section 5.2 with the same column structure as Tables 1 and 2.
- NVML and "hardware-validated" are both defined on first use.
- Page count reported. Baseline is 14; the limit is 15. If your additions push it past 15,
  say so explicitly rather than trimming content silently.

### Report back
Page count before and after (baseline 14); which CaFo PNG turned out to be Accuracy versus
Loss; **which of the 15 missing bibliography entries you could verify and which you could
not**; the T11 discrepancy; any thesis text you could not adapt cleanly to LNCS style; and
how you worded the environment paragraph in task 4d.

---
## PHASE 2 — Fairness infrastructure (BLOCKS ALL EXPERIMENTS)

<SHARED CONTEXT>

### Why this phase exists
Audit of the configs proved three claims in Section 4 of the paper are unsupported:
- "Consistent Early Stopping": BP stops on `bp_val_loss` with min_delta 0.0 and patience
  varying 5..40 per config; FF stops on `FF_Hinton/Val_Acc_Epoch` (ACCURACY) with
  min_delta 0.01 and patience 20; CaFo stops on `val_loss` with patience 5..10 and
  min_delta 0.0005-0.001; MF uses `mf_early_stopping_patience` 4/5/10 with min_delta
  0.0001. On the CIFAR-10 3x2000 headline config MF has patience 4 while BP has 20 —
  patience directly controls training duration, hence the reported time and energy.
- "Optuna applied to all algorithms": BP tuning uses a Median pruner while every
  configs/tuning/{ff,cafo,mf}_* sets `pruner: None`, so BP's 50 nominal trials include
  truncated ones while forward-only methods get 50 complete ones.
- Four sets of byte-identical hyperparameters exist across supposedly independent
  searches, because the TPE sampler is seeded from `general.seed` (42) for every study.

### Preconditions
None, but NO EXPERIMENT MAY RUN until this phase is merged and its tests pass.

### Read first
configs/base.yaml; a representative config from each of configs/{bp_baselines,ff,cafo,mf};
all of configs/tuning/; src/training/engine.py; src/utils/monitoring.py;
src/algorithms/mf.py; scripts/run_optuna_search.py;
scripts/slurm_scripts/run_array.slurm.

### Tasks
1. HARMONISE EARLY STOPPING. Introduce a single early-stopping block in configs/base.yaml
   — one metric (validation loss), min_delta 0.0, one patience value, one max-epoch cap —
   applied identically to every algorithm within a configuration group. Remove the
   per-algorithm keys (`bp_val_loss`, `FF_Hinton/Val_Acc_Epoch`,
   `predictor_early_stopping_metric`, `mf_early_stopping_patience`) in favour of the
   shared keys, updating the code that reads them. Keep the per-algorithm plumbing only
   where an algorithm genuinely has a different training loop shape (e.g. CaFo's
   per-predictor stopping) and document that in one line.
2. HARMONISE THE PRUNER to `pruner: None` for ALL algorithms across configs/tuning/.
   Offset the cost by introducing a reduced HPO epoch budget applied identically to every
   algorithm (a single `tuning.max_epochs` key). Do NOT use Median-for-all: truncating MF
   mid-layer is not the same event as truncating BP mid-epoch, so intermediate values are
   not comparable across algorithms.
3. FIX THE OPTUNA SEED COLLISION. In scripts/run_optuna_search.py (~lines 132-143) the TPE
   sampler is seeded from `general.seed`. Derive it instead from a stable hash of
   (algorithm_name, dataset_name, architecture_id) so studies are independent yet
   reproducible. Log the derived seed.
4. EQUALISE SEARCH SPACES. Add `weight_decay` to the MF space (currently pinned at 0.0).
   Widen FF's `ff_learning_rate` from its ~1.4-decade range to BP's 3 decades
   (1e-5 to 1e-2). Leave algorithm-specific parameters alone; the goal is comparable
   breadth, not identical parameter counts.
5. FIX THE BP BASELINE ASYMMETRY. In src/training/engine.py, `_create_mlp_model`
   (~lines 68-73) builds `MF_MLP(**arch_params)` for BOTH the MF and BP paths —
   `is_bp_baseline` is ignored — so the MF-family BP baseline carries unused
   `projection_matrices` (90,720 parameters on CIFAR-10 3x2000) that receive AdamW
   optimiser state and weight decay. Mirror the treatment already used for the FF family
   at ~lines 44-66, which builds a fresh `nn.Sequential`. This is reviewer R2's comment
   verbatim: "the paper claimed that the BP baselines were identical but algorithm
   specific components were removed."
6. EXTEND THE MONITOR. src/utils/monitoring.py currently samples only
   `nvmlDeviceGetPowerUsage` (~L134) and `nvmlDeviceGetMemoryInfo` (~L164). Add
   `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetTemperature`, `nvmlDeviceGetClockInfo`,
   `nvmlDeviceGetComputeRunningProcesses`, AND process RSS via `psutil`. Emit a per-run
   time-series CSV alongside the existing W&B logging. Keep the existing 0.2 s sampling
   interval (`monitoring.energy_interval_sec`) and the trapezoidal energy integration.
   RSS is mandatory: Phase 3 introduces a host-side activation cache, and without RSS it
   would appear to cost nothing.
7. IMPLEMENT THREE MF ACTIVATION-CACHE STRATEGIES behind one config flag in
   src/algorithms/mf.py. The current code (~lines 418-423) recomputes layers 0..i-1 under
   `torch.no_grad()` from the raw input for every batch of every epoch while training
   layer i — an O(L^2) forward cost. Add:
     - `recompute`  : current behaviour, unchanged, the default
     - `cache_device`: cache layer i-1 activations on the GPU
     - `cache_host` : cache in CPU RAM, stream per batch
   Caching is sound ONLY because detachment plus layer-sequential training freeze layers
   0..i-1 while layer i trains. Encode that precondition as an assertion so the flag can
   never be enabled for a joint-gradient algorithm.
   Caching must be mathematically identical to recomputation — same values, computed once.
8. FIX THE SLURM ARRAY. scripts/slurm_scripts/run_array.slurm declares `--array=1-25` but
   CONFIG_FILES holds 27 entries (11 BP + 8 CaFo + 4 FF + 4 MF), so the last two MF
   configs never ran. It also references configs/ff/mnist_mlp_3x1000_SGD_ref.yaml, which
   does not exist (the real file is mnist_mlp_3x1000_SGD.yaml). Fix both, and convert to a
   2-D array over (config x seed). Seed scaling needs NO Python change: engine.py ~L177
   already honours the `EXPERIMENT_SEED` environment variable as an override of
   `general.seed`.
9. WRITE tests/test_fairness_invariants.py asserting across all configs that, within each
   configuration group: early-stopping metric, min_delta, patience and max-epoch cap are
   identical across algorithms; pruner policy is uniform; batch size and train/val split
   are identical; and no two configs share tuned hyperparameter values unintentionally
   (allow an explicit opt-out list for genuine intentional sharing). This converts
   fairness from a prose claim into a CI check.

### Constraints
- Do NOT change any algorithm's mathematics. Only protocol, instrumentation and plumbing.
- Do NOT delete the original hyperparameter values — Phase 4 needs them as a fallback and
  Phase 6 needs them for the disclosure table. Preserve them (e.g. under a
  `legacy_hyperparameters:` key or a committed snapshot).
- Do NOT touch anything under latex_source/.
- The FF training path currently works; do not regress it while refactoring the shared
  early-stopping keys.

### Acceptance criteria
- `rtk pytest tests/` passes, including the new test_fairness_invariants.py.
- A short smoke run of each of BP, FF, CaFo and MF completes and produces a per-run CSV
  containing power, GPU memory, utilisation, temperature, SM clock and process RSS.
- The three MF cache strategies produce identical loss trajectories at a fixed seed
  (modulo GPU nondeterminism). Demonstrate this with a short two-layer run.
- The cache precondition assertion fires if the flag is set on a joint-gradient path.
- The SLURM array launches exactly n_configs x n_seeds jobs in a dry run.

### Report back
The full early-stopping settings table you replaced (metric, min_delta, patience per
algorithm per config) — Phase 6 needs it verbatim for the disclosure table. The measured
wall-time delta between `recompute` and the two cache strategies on one config. Peak GPU
memory and peak RSS for all three strategies.

---
## PHASE 3 — Ablation ladder: implementation and critical experiments

<SHARED CONTEXT>

### Why this phase exists
Two reviewers demand it. R3: "Because Mono-Forward places a cross-entropy loss at every
hidden layer, it inherently behaves like a deep supervision network. It would be great if
you could add a standard backpropagation baseline that uses matching layer-wise auxiliary
losses. This is the ONLY way to prove whether MF's efficiency and accuracy gains stem from
its unique forward-only mechanism or simply from localised objective constraints."
R5: "A stronger comparison would include additional baselines, such as deep-supervision or
local-loss backpropagation variants."

### The ladder (six rungs, four MLP configurations, n = 20 seeds)
| # | Rung | Readout | Aux losses | Gradients | Can cache | Isolates |
|1| BP | output_layer | none | global | no | reference |
|2| BP-DS | output_layer | each layer via M_i | global | no | auxiliary supervision |
|3| MF-Joint | M_L | each layer via M_i | global | no | readout / extra params |
|4| MF-recompute | M_L | each layer via M_i | local, sequential | - | locality, direct |
|5| MF-cache-device | M_L | each layer via M_i | local, sequential | yes | enabled optimisation |
|6| MF-cache-host | M_L | each layer via M_i | local, sequential | yes | same, GPU-neutral |
Rungs 1-3 cannot cache: joint gradients make any cached activation stale within one step.
Rung 4 is the accepted paper's implementation, which keeps the rung-3 comparison honest.

### Preconditions
Phase 2 must be merged and its tests green. In particular the harmonised early stopping,
the `pruner: None` policy, the per-study Optuna seed, the BP baseline symmetry fix and the
three cache strategies must all be in place.

### Read first
src/algorithms/mf.py (especially `mf_local_loss_fn` and the layer loop ~L418-423);
src/architectures/mf_mlp.py (`self.layers`, `self.projection_matrices` ~L89,
`self.output_layer` ~L80, `forward_with_intermediate_activations` ~L138);
src/training/engine.py (`get_model_and_adapter` ~L129, training-function dispatch
~L442-443); src/baselines/ ; src/tuning/optuna_objective_mf.py;
scripts/run_optuna_search.py (dispatch ~L72).

### Tasks — DO TASK 1 FIRST, IN ISOLATION
1. WRITE THE GRADIENT TEST BEFORE ANYTHING ELSE. Assert that layer-0 weights receive a
   non-zero gradient from the layer-L auxiliary loss when gradients are allowed to flow
   jointly. If `forward_with_intermediate_activations` in src/architectures/mf_mlp.py
   detaches internally, then BP-DS silently collapses onto MF, the ladder measures
   nothing, and the item two reviewers called essential is void. If this test cannot be
   made to pass, STOP and report — the ladder needs a different implementation path.
2. ADD src/baselines/bp_ds.py with `train_bp_ds_model` and `evaluate_bp_ds_model`,
   mirroring the signatures of the existing `train_bp_model` / `evaluate_bp_model`.
   Import `mf_local_loss_fn` from src/algorithms/mf.py VERBATIM — do not reimplement it —
   so the auxiliary loss is provably identical to MF's. Signature reminder:
   `mf_local_loss_fn(activation_i, projection_matrix_i, targets, criterion)` computing
   `goodness = a_i @ M_i.t()` then cross-entropy.
   Expose an `aux_weight` hyperparameter controlling the auxiliary-loss contribution.
3. ADD the MF-Joint path: readout via M_L, auxiliary losses at every layer, but joint
   gradients and no detachment. This is the strict single-variable neighbour of MF.
4. REGISTER both in src/algorithms/__init__.py and branch appropriately in
   `get_model_and_adapter` (src/training/engine.py ~L129, where
   `is_bp_baseline = algorithm_name == "bp"`).
5. ADD src/tuning/optuna_objective_bpds.py (and the MF-Joint equivalent) following the
   existing optuna_objective_mf.py pattern, including `aux_weight` in the search space.
   Wire into the dispatch in scripts/run_optuna_search.py (~L72).
6. ADD configs: configs/bp_ds/ and configs/mf_joint/ for the four MLP configurations
   (mnist_mlp_2x1000, mnist_mlp_3x1000 or 4x2000, fashion_mnist_mlp_2x1000,
   cifar10_mlp_3x2000 — match whatever the existing MF configs cover), plus
   configs/tuning/bpds_*_tune.yaml and mfjoint_*_tune.yaml.
7. TUNE FOUR RUNGS INDEPENDENTLY: BP, BP-DS, MF-Joint, MF — 16 studies, 50 trials each,
   `pruner: None`, reduced HPO epoch budget. Rungs 5 and 6 MUST SHARE RUNG 4's
   hyperparameters: caching is mathematically identical to recomputation, so the
   optimisation trajectory is unchanged and a separate search would be meaningless.
   Estimated cost ~13 GPU-h.
8. RUN THE LADDER AT n = 20: 4 configs x 6 rungs x 20 seeds = 480 runs, ~27 GPU-h, via the
   2-D SLURM array from Phase 2. Twenty seeds is chosen so the INTERNAL comparison
   "does BP-DS equal MF?" is decidable — that comparison is the trigger for reframing the
   paper's claims, and at n = 7 it could not distinguish "equal" from "underpowered".
9. PRE-REGISTER THE ANALYSIS BEFORE LOOKING AT RESULTS. Write the analysis script first.
   Primary test paired-by-seed (the shared seed set fixes the train/val split and data
   ordering, so pairing removes real nuisance variance). Welch rather than Student —
   variances are visibly unequal, badly so on timing. Holm correction within each metric
   family. Bootstrap 95% CIs reported alongside every p-value. For every PARITY claim use
   TOST equivalence testing with a pre-specified margin of 0.25 percentage points.
   A non-significant p-value must NEVER be reported as evidence of equivalence.
10. RUN TWO DIAGNOSTIC CONDITIONS ON CIFAR-10 ONLY (~1 GPU-h): harmonised early stopping
    with rung 4, and the ORIGINAL early stopping with rung 5. Harmonisation is expected to
    reduce MF's advantage while the cache fix increases it; without this attribution the
    two effects are entangled and the headline change is uninterpretable.
11. TWO-STAGE SIGMA CHECK. After task 8, recompute the pooled SD from 20 real observations
    and top up seeds only if it demands more. Every published effect size derives from
    n = 3, where the SD's 95% CI spans roughly [0.5, 5.7] x sigma-hat, and the paper and
    thesis disagree by 5.6x on the CIFAR-10 SD (0.28 vs 0.05). This resolves that
    discrepancy automatically.

### Constraints
- Do NOT modify `mf_local_loss_fn`. Import it. If it must change, the ladder is invalid.
- Do NOT enable a cache strategy on rungs 1-3. The Phase 2 assertion should prevent it;
  verify that it does.
- Do NOT peek at ladder results before the analysis script is written and committed.
- Do NOT change the four MLP architectures.

### Acceptance criteria
- The gradient test passes and is committed BEFORE any cluster job is submitted.
- Rungs 4, 5 and 6 agree on final accuracy to within run-to-run GPU nondeterminism at a
  fixed seed. Any real deviation is a cache bug — this is the single best correctness
  check for the Phase 2 cache work.
- All 480 runs complete; no silent array truncation.
- The analysis script emits, for every configuration: mean +/- bootstrap 95% CI per rung
  per metric, paired per-seed differences, Welch + Holm p-values, and TOST verdicts at
  0.25 pp.

### Report back
The per-rung decomposition of the BP -> MF-cache energy and time saving (how much comes
from auxiliary supervision, readout, locality, caching). The TOST verdict for every parity
claim. The recomputed sigma versus the published 0.28 and 0.05. Whether BP-DS is
statistically equivalent to MF — this determines whether the paper's framing must change.

---
## PHASE 4 — Remaining experimental matrix

<SHARED CONTEXT>

### Preconditions
Phase 2 merged and green. Phase 3 may run concurrently — this phase touches different
configs and does not depend on the ladder.

### Seed policy
n = 7 for FF and the remaining BP MLP baselines. n = 5 for CaFo and the CNN baselines.
(The ladder is n = 20 and is Phase 3's responsibility.)
Rationale: two reviewers called three runs insufficient. FF's claims are large-effect
superiority (d ~ 6.6 on accuracy, ~13x on time), so 7 is ample there.

### Tasks
1. Re-tune and re-run FF at n = 7. Four configs. The 13x-slower claim is a headline and is
   affected by the early-stopping metric mismatch fixed in Phase 2 (FF previously stopped
   on ACCURACY with min_delta 0.01 and patience 20, a far laxer rule than BP's loss-based
   min_delta 0.0). Expect the gap to move. ~13 GPU-h tuning + ~5 GPU-h finals.
2. Re-tune and re-run CaFo: 8 configs (4 Rand-CE + 4 DFA-CE), 30 trials, n = 5.
   ~24 GPU-h tuning + ~10 GPU-h finals. This replaces the thesis table back-ported in
   Phase 1.
   NOTE: three CaFo configs were never independently tuned — cafodfa_cifar100_cnn_3block
   is byte-identical to cafodfa_cifar10_cnn_3block across all four hyperparameters, and
   cafo/cifar100_cnn_3block shares predictor_lr and predictor_weight_decay with
   cafo/cifar10_cnn_3block. The Phase 2 per-study Optuna seed fix should prevent
   recurrence; verify the new values differ.
3. Re-tune and re-run the BP CNN baselines and the remaining BP MLP baselines at n = 5-7.
   ~12 GPU-h tuning + ~6 GPU-h finals.
   NOTE: bp_baselines/mnist_mlp_4x2000_bp.yaml is byte-identical to
   mnist_mlp_3x1000_bp.yaml (lr 0.0001329291894316216, wd 0.0007114476009343421) — the
   4x2000 baseline was never independently tuned. Verify the new values differ.
4. Export the historical W&B run histories from project
   `przspyra11/BeyondBackpropagation` via the wandb API into a local archive. These are
   NOT a data source for the camera-ready — they serve two verification purposes:
   confirming that the paper's existing hardware traces are W&B system panels rather than
   NVML, and cross-checking that the from-scratch pipeline reproduces the published
   numbers.
5. REPRODUCTION CHECK: re-run ONE published configuration under the ORIGINAL (pre-Phase-2)
   settings and confirm it reproduces the accepted paper's numbers within CI. This proves
   the from-scratch pipeline is not silently different from the one that produced the
   accepted results. If it does not reproduce, STOP and report before proceeding.

### Constraints
- Use the Phase 2 harmonised protocol for all new runs. The only exception is task 5,
  which deliberately uses the legacy settings.
- Do NOT re-tune the ladder rungs — that is Phase 3's job and duplicating it wastes
  ~13 GPU-h.
- Respect the 8-hour SLURM wall clock; chunk the tuning campaigns accordingly.

### Fallback if the A100 queue is contended
HPO is the bottleneck (~62 GPU-h across Phases 3 and 4, on the critical path in weeks
2-4). The designated cut is CaFo re-tuning (~24 GPU-h): keep the thesis hyperparameters
for CaFo only, disclosed in a footnote. CaFo is not a headline claim and the Phase 1
back-ported table already satisfies reviewer R3b. Do NOT cut FF or the ladder.

### Acceptance criteria
- Every config's new hyperparameters differ from every other config's, except where
  intentional and recorded in the test_fairness_invariants.py opt-out list.
- All runs emit the extended per-run CSV from Phase 2.
- The reproduction check in task 5 lands within CI of the published numbers.

### Report back
Old versus new hyperparameters for every re-tuned config. The reproduction-check result.
How much FF's time gap moved after early-stopping harmonisation.

---
## PHASE 5 — Plotting pipeline

<SHARED CONTEXT>

### Why this phase exists
Reviewer R1: "the plots in Fig. 3 and Fig. 4 are completely illegible and don't convey
what the authors are trying to explain in the paper text."
Diagnosis: Figure 3 sits at 0.48\linewidth (~5.9 cm on a ~12.2 cm LNCS textwidth) from a
~1568 px source with ~20 px glyphs, giving roughly 2.5 pt effective type. Figure 4 sits at
0.32\linewidth, giving roughly 1.7 pt. A seven-entry legend consumes ~25% of each panel,
three near-identical blues fail in greyscale, in-image titles duplicate the LaTeX captions,
and Figure 4b's temperature axis spans 23-30 C so a 3 C difference looks dramatic.
There is NO plotting code anywhere in the repository. Every current figure is a Weights &
Biases UI export, which is why none can be regenerated at higher resolution.

### Preconditions
Phase 2's extended monitor must be emitting per-run CSVs. Phase 3 and 4 results are needed
for the final figures, but the data layer and style module can be built in parallel with
them using smoke-run data.

### Tasks
1. BUILD THE DATA LAYER. The Phase 2 monitor emits a per-run time series (timestamp, power,
   GPU memory, utilisation, temperature, SM clock, process RSS). Add a per-run summary
   record: config, algorithm, cache strategy, dataset, architecture, seed, test accuracy,
   wall time, energy, peak GPU memory, peak RSS, epochs run, stop reason. Write an
   aggregator producing ONE tidy long-format table that every figure AND every table in
   the paper draws from, so no number is ever transcribed by hand.
2. BUILD THE STYLE MODULE — a single rcParams configuration enforcing:
   - Generation at exact final size (LNCS textwidth ~122 mm) so \includegraphics never
     scales the output. Compute panel widths from the fraction used in main.tex.
   - 8 pt base font, 7 pt floor for tick labels.
   - Okabe-Ito colourblind-safe palette.
   - Linestyle AND marker cycling so every panel survives greyscale printing.
   - No in-image titles. Captions live in LaTeX only.
   - Legends outside the axes, or one legend shared across a figure.
   - Zero-based axes, or an explicit axis break where zero-basing is meaningless.
   - Vector PDF output.
3. THE LADDER WATERFALL (new, intended as the paper's centrepiece). Decompose the total
   BP -> MF-cache energy saving across the five rung transitions, showing how much each of
   auxiliary supervision, readout choice, gradient locality and activation caching
   contributes. This is the direct visual answer to reviewer R3's "only way to prove".
4. THE TIME-MEMORY FRONTIER (new). A scatter with BP as a single reference point and MF
   rungs 4, 5 and 6 tracing the frontier: GPU memory on one axis, wall time on the other,
   with host RSS carried by marker size or a companion panel. Three points make a frontier;
   two make a line segment — plot all three.
5. THE EQUIVALENCE FOREST PLOT (new). Per-configuration mean accuracy differences with
   bootstrap 95% CIs and the +/- 0.25 pp equivalence band shaded. This is how a parity
   claim should be shown, and it answers reviewer R5's "some of the reported accuracy
   differences are modest".
6. REBUILD THE FOUR EXISTING FIGURES from the new data: FF (2 panels), CaFo (2 panels,
   with the labels correct this time), MF (2 panels, with an NVML power trace REPLACING
   the temperature panel), and MF-vs-BP convergence plotted against WALL-CLOCK TIME rather
   than epochs. Use bootstrap CI bands over the available seeds rather than +/- SD ribbons.
7. DIAGNOSTIC PLOTS, not for the paper: the early-stopping and cache-strategy attribution
   from Phase 3 task 10, for use in the response-to-reviewers letter.

### Critical correctness note
The paper currently attributes all hardware traces to NVML, but three of the four come
from W&B system panels — "Process Memory In Use" is a host-RSS metric in MB, which is why
FF_4.png shows BP at ~815-925 units while Table 1 reports an NVML BP peak of
1168 +/- 4 MiB. Every regenerated figure must draw from the Phase 2 NVML CSVs, and every
caption must name the actual instrument and unit.

### Constraints
- Do NOT hand-edit any figure. Everything is generated from the tidy table.
- Do NOT reuse any W&B UI export in the camera-ready.
- Keep the plotting code in scripts/ or a small src/plotting/ package; do not scatter
  matplotlib calls through the training code.

### Acceptance criteria
- Every figure renders at >= 8 pt type when inspected at 100% zoom at its final size.
- Every figure is legible in greyscale.
- Every number in every paper table traces back to the aggregated tidy table.
- Re-running the plotting script reproduces byte-identical PDFs from the same inputs.

### Report back
A before/after legibility comparison for Figures 3 and 4. Any metric the paper reports
that has no corresponding column in the tidy table.

---
## PHASE 6 — Manuscript revision and response-to-reviewers letter

<SHARED CONTEXT>

### Preconditions
Phases 1, 3, 4 and 5 complete. You need final numbers and final figures.

### Read first
latex_source/review.tex (all five reviews, in full); latex_source/PPAM_2026_SUBMISSION/
main.tex; the Phase 2 report containing the original early-stopping settings table; the
Phase 3 analysis output; the Phase 4 old-versus-new hyperparameter tables.

### Reviewer traceability — every one of these must be closed and cited in the letter
R1a condensed technical report, poor readability | R1b Figures 3 and 4 illegible |
R2a Section 6 lacks detail and analysis, "seemed preliminary" | R2b architecture details
missing | R2c BP baseline component removal may itself explain the differences |
R2d library maturity favours BP | R3a NVML never defined | R3b CaFo has no table |
R3c deep-supervision BP baseline, "the only way to prove" | R4a "hardware-validated"
unclear | R4b only three runs | R5a evaluative rather than algorithmic | R5b small
datasets, shallow architectures, flattened CIFAR | R5c three runs, modest differences |
R5d deep-supervision or local-loss variants | R5e single GPU platform | R5f clarify
limitations, strengthen baselines and statistics.

### Tasks
1. REWRITE SECTION 4 FAIRNESS PROTOCOLS 2, 3 AND 4. Protocol 4's "consistent early
   stopping" and protocol 3's "Optuna applied to all algorithms for every configuration"
   are both unsupported by the configs as submitted. Add:
   a. An early-stopping settings table showing what was ACTUALLY used in the submitted
      version versus the harmonised protocol now used. Phase 2 reports the original values.
   b. A search-space table (from thesis Appendix A plus the Phase 2 widenings), restating
      protocol 3 as "equal trial budget, algorithm-appropriate search spaces".
   c. Disclosure of the pruner asymmetry: BP used a Median pruner while FF, CaFo and MF had
      pruning disabled, so BP's nominal 50 trials included truncated ones. Note explicitly
      that this asymmetry DISFAVOURED BP — that is, it worked in the paper's favour — and
      that it has now been removed.
   d. Disclosure of the three configs whose hyperparameters were transplanted rather than
      independently searched, and their re-tuned replacements.
2. ADD THE ABLATION-LADDER SUBSECTION to Section 5, presenting the time-memory frontier as
   a CONSEQUENCE of gradient locality rather than an implementation note: BP, BP-DS and
   MF-Joint cannot cache activations because joint gradients make any cached activation
   stale within one step, whereas MF can because detachment plus layer-sequential training
   freeze the preceding layers. This addresses R5a's "mainly evaluative rather than
   algorithmic" — the frontier is a structural property, not an evaluation.
3. ANSWER R2d EMPIRICALLY, not rhetorically. R2 hypothesised that BP looks better because
   libraries are better optimised for it. That hypothesis was CORRECT in a specific and
   measurable way: the original MF implementation recomputed the forward pass O(L^2) times.
   Report the measured effect of fixing it. Be explicit that the previously published MF
   advantage was UNDERSTATED for this reason.
4. EXPAND SECTION 6 (R1a, R2a). The reviewers found it thin and preliminary. Add the
   per-rung attribution, the frontier discussion, and the hardware-utilisation finding —
   mf_3.png shows BP at 16-17% and MF at 9-11% GPU utilisation, meaning both workloads are
   launch-bound rather than compute-bound, which is a substantive observation the current
   text does not make.
5. UPDATE SECTION 6.4 LIMITATIONS: remove the "three runs are insufficient" item (now
   resolved); add the achieved equivalence margin and the minimum detectable effect; add
   the GPU-ONLY ENERGY ACCOUNTING GAP (NVML cannot see the CPU and PCIe energy that the
   host-cache variant shifts off-instrument); retain and sharpen R5b's scope limits (small
   datasets, shallow architectures, flattened CIFAR inputs) and R5e's single-platform caveat.
6. CORRECT THE INSTRUMENT ATTRIBUTION throughout. Every caption must name the instrument
   that actually produced the trace.
7. WRITE THE RESPONSE LETTER. One numbered entry per reviewer comment, each pointing to a
   specific section, table or figure. Additionally and voluntarily disclose the defects
   found by our own audit that NO reviewer raised: the Figure 3 subfloat label swap, the
   early-stopping inconsistency, the pruner asymmetry, the transplanted hyperparameters,
   and the instrument misattribution. The repository is public and cited in the paper, so
   all of these are discoverable by any reader after publication. Disclosing them
   voluntarily is far stronger than being caught.

### Constraints
- LNCS style, UK English, existing macros (\FF{}, \MF{}, \BP{}, \CaFo{}, \CaFoDFA{}),
  siunitx units, booktabs tables.
- 15 pages maximum. If over: move the search-space table and the full environment listing
  to the public repository and cite by URL from the Code Repository section.
- NO fabricated numbers. Every figure in the text must trace to the Phase 5 tidy table.
- If the ladder shows BP-DS statistically equivalent to MF, the abstract MUST be reframed
  from "locality causes the saving" to an attributed decomposition. This reframing is
  pre-approved; do not soften the finding to preserve the original claim.

### Acceptance criteria
- All 17 reviewer items above have a numbered response and a manuscript location.
- Compiles with llncs.cls at <= 15 pages, zero undefined references, zero missing citations.
- No claim in the paper contradicts the repository as published.

### Report back
Final page count. Any reviewer item you could not fully close and why. Any place where the
new results contradict the accepted version's claims.
