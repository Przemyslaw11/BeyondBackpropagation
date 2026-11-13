# W1 — attribution ladder (PREREG-phase2.md §3/§4)

Reduced 440 runs from `/Users/pspyra/Desktop/BeyondBackpropagation_ready/jpdc_data/w1/analysis_input_20261111`; seeds 42–61; primary dataset `cifar100`; CIFAR-10 is the consistency check.
Passes from JSON `epochs_completed`; energy from each run's exact `monitoring_csv_path`; wall from `training_duration_sec`. Paired Wilcoxon + 10,000-draw bootstrap; Holm over {H-W1a, H-W1b, H-W1c} on CIFAR-100 only. Missing seeds are never imputed.

## Missing seeds and not-run arms

- Not run: MF-Seq cap-300 / cifar10, MF-Seq cap-300 / cifar100

## Per-arm endpoints (mean [95% CI])

| arm | dataset | n | acc % | passes | energy Wh | training wall s |
|---|---|---|---|---|---|---|
| BP | cifar10 | 20 | 62.47 [62.31, 62.63] | 52.95 [51.37, 54.53] | 5.16 [4.98, 5.35] | 255.41 [246.67, 264.16] |
| BP | cifar100 | 20 | 32.90 [32.77, 33.03] | 43.90 [42.91, 44.89] | 4.34 [4.22, 4.45] | 212.86 [207.59, 218.13] |
| MF-Joint | cifar10 | 20 | 62.28 [62.08, 62.48] | 51.30 [49.59, 53.01] | 5.10 [4.92, 5.28] | 252.11 [242.72, 261.50] |
| MF-Joint | cifar100 | 20 | 34.76 [34.59, 34.93] | 50.25 [49.16, 51.34] | 5.15 [5.01, 5.29] | 246.33 [240.48, 252.19] |
| MF-Simul | cifar10 | 20 | 61.61 [61.37, 61.84] | 58.25 [56.19, 60.31] | 5.72 [5.52, 5.93] | 286.10 [276.02, 296.19] |
| MF-Simul | cifar100 | 20 | 32.63 [32.42, 32.84] | 35.70 [34.70, 36.70] | 3.54 [3.42, 3.66] | 174.61 [168.95, 180.28] |
| MF-Seq | cifar10 | 20 | 66.37 [66.24, 66.51] | 211.20 [204.72, 217.68] | 18.05 [17.53, 18.58] | 1032.68 [993.63, 1071.72] |
| MF-Seq | cifar100 | 20 | 37.94 [37.78, 38.10] | 191.35 [187.27, 195.43] | 16.38 [15.83, 16.92] | 932.32 [900.81, 963.83] |
| Seq-Global | cifar10 | 20 | 65.20 [65.02, 65.38] | 234.15 [229.16, 239.14] | 20.93 [20.44, 21.43] | 1136.38 [1109.11, 1163.65] |
| Seq-Global | cifar100 | 20 | 38.84 [38.70, 38.97] | 258.40 [251.40, 265.40] | 23.80 [22.99, 24.61] | 1281.96 [1228.77, 1335.16] |
| MF-Seq-nom0 | cifar10 | 20 | 66.44 [66.24, 66.63] | 160.15 [158.35, 161.95] | 13.68 [13.37, 14.00] | 779.73 [764.54, 794.91] |
| MF-Seq-nom0 | cifar100 | 20 | 37.93 [37.70, 38.17] | 147.10 [142.96, 151.24] | 12.71 [12.22, 13.20] | 713.44 [686.64, 740.25] |
| BP-fixed | cifar10 | 20 | 65.59 [65.43, 65.75] | 211.00 [211.00, 211.00] | 20.99 [20.67, 21.31] | 1036.96 [1018.40, 1055.52] |
| BP-fixed | cifar100 | 20 | 34.20 [34.04, 34.35] | 191.00 [191.00, 191.00] | 18.94 [18.62, 19.26] | 929.70 [918.90, 940.51] |
| BP-bestval | cifar10 | 20 | 62.64 [62.51, 62.78] | 52.95 [51.37, 54.53] | 5.27 [5.09, 5.46] | 258.50 [249.31, 267.69] |
| BP-bestval | cifar100 | 20 | 32.88 [32.76, 33.00] | 43.90 [42.91, 44.89] | 4.38 [4.28, 4.48] | 214.33 [209.25, 219.42] |
| BP-sched | cifar10 | 20 | 65.53 [65.42, 65.65] | 100.00 [100.00, 100.00] | 9.97 [9.76, 10.17] | 492.40 [481.07, 503.73] |
| BP-sched | cifar100 | 20 | 34.85 [34.71, 34.99] | 100.00 [100.00, 100.00] | 9.91 [9.78, 10.04] | 486.54 [480.25, 492.83] |
| MF-Seq Protocol B | cifar10 | 20 | 61.66 [61.51, 61.81] | 56.00 [56.00, 56.00] | 4.69 [4.64, 4.74] | 265.85 [264.54, 267.15] |
| MF-Seq Protocol B | cifar100 | 20 | 32.92 [32.79, 33.04] | 44.00 [44.00, 44.00] | 3.70 [3.65, 3.76] | 210.22 [208.43, 212.00] |
| nom0 Protocol B | cifar10 | 20 | 62.48 [62.35, 62.61] | 54.00 [54.00, 54.00] | 4.59 [4.55, 4.63] | 258.24 [256.46, 260.01] |
| nom0 Protocol B | cifar100 | 20 | 34.03 [33.86, 34.20] | 45.00 [45.00, 45.00] | 3.80 [3.76, 3.85] | 215.31 [214.07, 216.55] |

## Per-arm medians (PREREG requires mean and median)

| arm | dataset | n | acc % | passes | energy Wh | training wall s |
|---|---|---|---|---|---|---|
| BP | cifar10 | 20 | 62.44 | 52.5 | 5.12 | 255.0 |
| BP | cifar100 | 20 | 32.86 | 43.5 | 4.27 | 210.9 |
| MF-Joint | cifar10 | 20 | 62.17 | 51.0 | 5.12 | 247.1 |
| MF-Joint | cifar100 | 20 | 34.84 | 50.5 | 5.14 | 246.0 |
| MF-Simul | cifar10 | 20 | 61.58 | 59.0 | 5.80 | 286.3 |
| MF-Simul | cifar100 | 20 | 32.61 | 35.5 | 3.54 | 174.6 |
| MF-Seq | cifar10 | 20 | 66.40 | 207.0 | 17.76 | 1016.3 |
| MF-Seq | cifar100 | 20 | 37.99 | 190.5 | 16.10 | 919.8 |
| Seq-Global | cifar10 | 20 | 65.18 | 231.5 | 21.02 | 1131.4 |
| Seq-Global | cifar100 | 20 | 38.82 | 260.5 | 23.15 | 1247.6 |
| MF-Seq-nom0 | cifar10 | 20 | 66.48 | 161.0 | 13.74 | 778.9 |
| MF-Seq-nom0 | cifar100 | 20 | 38.06 | 151.0 | 12.76 | 724.4 |
| BP-fixed | cifar10 | 20 | 65.58 | 211.0 | 20.68 | 1025.2 |
| BP-fixed | cifar100 | 20 | 34.27 | 191.0 | 18.81 | 919.3 |
| BP-bestval | cifar10 | 20 | 62.64 | 52.5 | 5.25 | 255.1 |
| BP-bestval | cifar100 | 20 | 32.87 | 43.5 | 4.33 | 211.8 |
| BP-sched | cifar10 | 20 | 65.55 | 100.0 | 9.78 | 480.9 |
| BP-sched | cifar100 | 20 | 34.85 | 100.0 | 9.90 | 482.4 |
| MF-Seq Protocol B | cifar10 | 20 | 61.73 | 56.0 | 4.69 | 265.4 |
| MF-Seq Protocol B | cifar100 | 20 | 32.95 | 44.0 | 3.67 | 209.5 |
| nom0 Protocol B | cifar10 | 20 | 62.38 | 54.0 | 4.58 | 258.3 |
| nom0 Protocol B | cifar100 | 20 | 34.01 | 45.0 | 3.77 | 215.7 |

## 2x2 attribution table

Cells: rows = loss attachment, columns = prefix gradients. Effects are paired-by-seed; positive loss attachment means every-stage loss has higher values than output-only loss; positive prefix gradient means trainable prefix has higher values than frozen/detached.

### cifar10 — accuracy (%)

| loss attachment | frozen/detached prefix | trainable prefix | main effect A | main effect P | interaction |
|---|---|---|---|---|---|
| output-only | MF-Simul: 61.61 (n=20) | MF-Joint: 62.28 (n=20) | +3.85 | -0.25 | -1.84 |
| every-stage | MF-Seq: 66.37 (n=20) | Seq-Global: 65.20 (n=20) |  |  |  |

**cifar10 / test_accuracy effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +3.846 | +3.693 | +3.999 | 0.0000 |
| prefix_gradients | 20 | -0.250 | -0.447 | -0.065 | 0.0296 |
| interaction | 20 | -1.840 | -2.236 | -1.422 | 0.0000 |

**cifar10 / passes effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +167.900 | +163.625 | +172.450 | 0.0001 |
| prefix_gradients | 20 | +8.000 | +4.449 | +11.625 | 0.0011 |
| interaction | 20 | +29.900 | +21.850 | +37.650 | 0.0002 |

**cifar10 / energy_wh effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +14.082 | +13.725 | +14.462 | 0.0000 |
| prefix_gradients | 20 | +1.128 | +0.751 | +1.498 | 0.0000 |
| interaction | 20 | +3.504 | +2.783 | +4.220 | 0.0000 |

**cifar10 / wall_sec effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +815.423 | +793.460 | +837.274 | 0.0000 |
| prefix_gradients | 20 | +34.857 | +10.353 | +58.605 | 0.0153 |
| interaction | 20 | +137.698 | +87.166 | +185.568 | 0.0002 |

### cifar100 — accuracy (%)

| loss attachment | frozen/detached prefix | trainable prefix | main effect A | main effect P | interaction |
|---|---|---|---|---|---|
| output-only | MF-Simul: 32.63 (n=20) | MF-Joint: 34.76 (n=20) | +4.69 | +1.51 | -1.23 |
| every-stage | MF-Seq: 37.94 (n=20) | Seq-Global: 38.84 (n=20) |  |  |  |

**cifar100 / test_accuracy effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +4.693 | +4.561 | +4.817 | 0.0000 |
| prefix_gradients | 20 | +1.513 | +1.361 | +1.673 | 0.0000 |
| interaction | 20 | -1.227 | -1.535 | -0.894 | 0.0000 |

**cifar100 / passes effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +181.900 | +177.125 | +186.501 | 0.0001 |
| prefix_gradients | 20 | +40.800 | +38.100 | +43.375 | 0.0001 |
| interaction | 20 | +52.500 | +46.550 | +57.900 | 0.0001 |

**cifar100 / energy_wh effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +15.745 | +15.191 | +16.320 | 0.0000 |
| prefix_gradients | 20 | +4.517 | +4.158 | +4.876 | 0.0000 |
| interaction | 20 | +5.817 | +5.110 | +6.501 | 0.0000 |

**cifar100 / wall_sec effects**

| effect | n | mean | 95% CI low | 95% CI high | raw p |
|---|---|---|---|---|---|
| loss_attachment | 20 | +896.666 | +865.502 | +929.353 | 0.0000 |
| prefix_gradients | 20 | +210.681 | +186.232 | +235.112 | 0.0000 |
| interaction | 20 | +277.923 | +227.325 | +329.657 | 0.0000 |

## Primary hypotheses

### cifar100 hypotheses

| hypothesis | status | n | contrast | delta | 95% CI | raw p | Holm p | note |
|---|---|---|---|---|---|---|---|---|
| H-W1a | not supported (threshold not met) | 20 | Seq-Global - MF-Seq AND Seq-Global - MF-Simul | +0.90 / +6.21 | [+0.69, +1.14] / [+5.97, +6.44] | 0.0000 | 0.0000 | threshold: <= -2.0 pp and <= +1.0 pp |
| H-W1b | supported | 20 | Seq-Global - MF-Seq | +0.90 | [+0.69, +1.14] | 0.0000 | 0.0000 | threshold: >= -1.0 pp |
| H-W1c | threshold met; not significant after Holm | 20 | MF-Seq-nom0 - MF-Seq | -0.00 (pass save 23%, energy save 22%) | [-0.31, +0.29] | 0.9405 | 0.9405 | threshold: acc >= -0.5 pp and pass/energy saving >= 15% |
| H-W1d | MF-Seq advantage survives | 20 | BP-fixed - MF-Seq | -3.74 (passes ratio 1.00) | [-3.95, -3.53] | 0.0000 | n/a | control; raw CI; passes ratio is descriptive (~1x design) |
| H-W1cap | not run | 0 | MF-Seq cap-300 - MF-Seq | n/a | [n/a, n/a] | n/a | n/a | conditional full arm; Step 1 pilot gates it |

### CIFAR-10 consistency (same thresholds; no Holm and no confirmatory support)

### cifar10 hypotheses

| hypothesis | status | n | contrast | delta | 95% CI | raw p | Holm p | note |
|---|---|---|---|---|---|---|---|---|
| H-W1a | not supported (threshold not met) | 20 | Seq-Global - MF-Seq AND Seq-Global - MF-Simul | -1.17 / +3.60 | [-1.40, -0.93] / [+3.30, +3.88] | 0.0001 | n/a | threshold: <= -2.0 pp and <= +1.0 pp |
| H-W1b | not supported (Seq-Global loss exceeds -1.0 pp) | 20 | Seq-Global - MF-Seq | -1.17 | [-1.40, -0.93] | 0.0001 | n/a | threshold: >= -1.0 pp |
| H-W1c | consistency threshold met; raw p not significant | 20 | MF-Seq-nom0 - MF-Seq | +0.06 (pass save 24%, energy save 24%) | [-0.11, +0.23] | 0.5217 | n/a | threshold: acc >= -0.5 pp and pass/energy saving >= 15% |
| H-W1d | compute-advantage framing retired (CI excludes worse than -1 pp) | 20 | BP-fixed - MF-Seq | -0.78 (passes ratio 1.00) | [-0.94, -0.62] | 0.0001 | n/a | control; raw CI; passes ratio is descriptive (~1x design) |
| H-W1cap | not run | 0 | MF-Seq cap-300 - MF-Seq | n/a | [n/a, n/a] | n/a | n/a | conditional full arm; Step 1 pilot gates it |

### cifar100 cost endpoints

Accuracy is test accuracy; passes, energy and wall are paired co-primary endpoints. Ratios are arm_a/arm_b; training wall excludes post-training probes.

| hypothesis | contrast | metric | n | mean diff | 95% CI | raw p | ratio [95% CI] |
|---|---|---|---|---|---|---|---|
| H-W1a | Seq-Global - MF-Seq | test_accuracy | 20 | +0.899 | [+0.695, +1.136] | 0.0000 | n/a |
| H-W1a | Seq-Global - MF-Seq | passes | 20 | +67.050 | [+61.450, +72.150] | 0.0001 | 1.351 [1.321, 1.380] |
| H-W1a | Seq-Global - MF-Seq | energy_wh | 20 | +7.425 | [+6.731, +8.111] | 0.0000 | 1.457 [1.412, 1.500] |
| H-W1a | Seq-Global - MF-Seq | wall_sec | 20 | +349.643 | [+300.420, +399.177] | 0.0000 | 1.379 [1.325, 1.431] |
| H-W1a | Seq-Global - MF-Simul | test_accuracy | 20 | +6.205 | [+5.968, +6.445] | 0.0000 | n/a |
| H-W1a | Seq-Global - MF-Simul | passes | 20 | +222.700 | [+216.100, +228.850] | 0.0001 | 7.260 [7.016, 7.498] |
| H-W1a | Seq-Global - MF-Simul | energy_wh | 20 | +20.262 | [+19.518, +21.072] | 0.0000 | 6.765 [6.439, 7.107] |
| H-W1a | Seq-Global - MF-Simul | wall_sec | 20 | +1107.347 | [+1060.055, +1158.304] | 0.0000 | 7.378 [7.021, 7.750] |
| H-W1b | Seq-Global - MF-Seq | test_accuracy | 20 | +0.899 | [+0.695, +1.136] | 0.0000 | n/a |
| H-W1b | Seq-Global - MF-Seq | passes | 20 | +67.050 | [+61.450, +72.150] | 0.0001 | 1.351 [1.321, 1.380] |
| H-W1b | Seq-Global - MF-Seq | energy_wh | 20 | +7.425 | [+6.731, +8.111] | 0.0000 | 1.457 [1.412, 1.500] |
| H-W1b | Seq-Global - MF-Seq | wall_sec | 20 | +349.643 | [+300.420, +399.177] | 0.0000 | 1.379 [1.325, 1.431] |
| H-W1c | MF-Seq-nom0 - MF-Seq | test_accuracy | 20 | -0.004 | [-0.307, +0.295] | 0.9405 | n/a |
| H-W1c | MF-Seq-nom0 - MF-Seq | passes | 20 | -44.250 | [-49.850, -39.000] | 0.0001 | 0.770 [0.744, 0.794] |
| H-W1c | MF-Seq-nom0 - MF-Seq | energy_wh | 20 | -3.666 | [-4.204, -3.167] | 0.0000 | 0.778 [0.749, 0.805] |
| H-W1c | MF-Seq-nom0 - MF-Seq | wall_sec | 20 | -218.874 | [-245.361, -193.345] | 0.0000 | 0.767 [0.742, 0.791] |
| H-W1d | BP-fixed - MF-Seq | test_accuracy | 20 | -3.743 | [-3.950, -3.529] | 0.0000 | n/a |
| H-W1d | BP-fixed - MF-Seq | passes | 20 | -0.350 | [-4.150, +3.350] | 0.9851 | 1.000 [0.981, 1.019] |
| H-W1d | BP-fixed - MF-Seq | energy_wh | 20 | +2.563 | [+1.944, +3.123] | 0.0000 | 1.162 [1.122, 1.198] |
| H-W1d | BP-fixed - MF-Seq | wall_sec | 20 | -2.615 | [-36.911, +25.461] | 0.5459 | 1.002 [0.969, 1.030] |

### cifar10 cost endpoints

Accuracy is test accuracy; passes, energy and wall are paired co-primary endpoints. Ratios are arm_a/arm_b; training wall excludes post-training probes.

| hypothesis | contrast | metric | n | mean diff | 95% CI | raw p | ratio [95% CI] |
|---|---|---|---|---|---|---|---|
| H-W1a | Seq-Global - MF-Seq | test_accuracy | 20 | -1.170 | [-1.400, -0.931] | 0.0001 | n/a |
| H-W1a | Seq-Global - MF-Seq | passes | 20 | +22.950 | [+15.950, +30.050] | 0.0003 | 1.113 [1.078, 1.148] |
| H-W1a | Seq-Global - MF-Seq | energy_wh | 20 | +2.880 | [+2.202, +3.568] | 0.0000 | 1.164 [1.123, 1.205] |
| H-W1a | Seq-Global - MF-Seq | wall_sec | 20 | +103.706 | [+56.226, +149.499] | 0.0006 | 1.107 [1.061, 1.154] |
| H-W1a | Seq-Global - MF-Simul | test_accuracy | 20 | +3.596 | [+3.304, +3.883] | 0.0000 | n/a |
| H-W1a | Seq-Global - MF-Simul | passes | 20 | +175.900 | [+171.400, +180.450] | 0.0001 | 4.039 [3.905, 4.182] |
| H-W1a | Seq-Global - MF-Simul | energy_wh | 20 | +15.210 | [+14.728, +15.684] | 0.0000 | 3.679 [3.537, 3.832] |
| H-W1a | Seq-Global - MF-Simul | wall_sec | 20 | +850.279 | [+824.972, +876.071] | 0.0000 | 3.993 [3.847, 4.145] |
| H-W1b | Seq-Global - MF-Seq | test_accuracy | 20 | -1.170 | [-1.400, -0.931] | 0.0001 | n/a |
| H-W1b | Seq-Global - MF-Seq | passes | 20 | +22.950 | [+15.950, +30.050] | 0.0003 | 1.113 [1.078, 1.148] |
| H-W1b | Seq-Global - MF-Seq | energy_wh | 20 | +2.880 | [+2.202, +3.568] | 0.0000 | 1.164 [1.123, 1.205] |
| H-W1b | Seq-Global - MF-Seq | wall_sec | 20 | +103.706 | [+56.226, +149.499] | 0.0006 | 1.107 [1.061, 1.154] |
| H-W1c | MF-Seq-nom0 - MF-Seq | test_accuracy | 20 | +0.061 | [-0.110, +0.226] | 0.5217 | n/a |
| H-W1c | MF-Seq-nom0 - MF-Seq | passes | 20 | -51.050 | [-57.600, -45.550] | 0.0001 | 0.761 [0.739, 0.780] |
| H-W1c | MF-Seq-nom0 - MF-Seq | energy_wh | 20 | -4.373 | [-4.927, -3.877] | 0.0000 | 0.760 [0.736, 0.782] |
| H-W1c | MF-Seq-nom0 - MF-Seq | wall_sec | 20 | -252.950 | [-288.510, -221.163] | 0.0000 | 0.759 [0.734, 0.781] |
| H-W1d | BP-fixed - MF-Seq | test_accuracy | 20 | -0.784 | [-0.939, -0.624] | 0.0001 | n/a |
| H-W1d | BP-fixed - MF-Seq | passes | 20 | -0.200 | [-6.650, +5.350] | 0.4322 | 1.003 [0.975, 1.028] |
| H-W1d | BP-fixed - MF-Seq | energy_wh | 20 | +2.935 | [+2.320, +3.518] | 0.0000 | 1.167 [1.130, 1.202] |
| H-W1d | BP-fixed - MF-Seq | wall_sec | 20 | +4.283 | [-36.501, +40.925] | 0.5706 | 1.010 [0.973, 1.043] |

### cifar100 descriptive controls (raw CIs; not in Holm)

| control | metric | n | mean diff | 95% CI | raw p | ratio [95% CI] |
|---|---|---|---|---|---|---|
| BP-bestval - BP | test_accuracy | 20 | -0.017 | [-0.182, +0.150] | 0.9133 | n/a |
| BP-bestval - BP | passes | 20 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.000 [1.000, 1.000] |
| BP-bestval - BP | energy_wh | 20 | +0.044 | [-0.030, +0.107] | 0.0897 | 1.011 [0.995, 1.026] |
| BP-bestval - BP | wall_sec | 20 | +1.474 | [-1.996, +4.332] | 0.0400 | 1.008 [0.992, 1.020] |
| BP-sched - BP | test_accuracy | 20 | +1.948 | [+1.793, +2.111] | 0.0000 | n/a |
| BP-sched - BP | passes | 20 | +56.100 | [+55.200, +57.000] | 0.0001 | 2.283 [2.236, 2.329] |
| BP-sched - BP | energy_wh | 20 | +5.575 | [+5.445, +5.705] | 0.0000 | 2.291 [2.238, 2.344] |
| BP-sched - BP | wall_sec | 20 | +273.681 | [+267.994, +279.106] | 0.0000 | 2.290 [2.244, 2.337] |
| BP-fixed - BP | test_accuracy | 20 | +1.297 | [+1.167, +1.435] | 0.0000 | n/a |
| BP-fixed - BP | passes | 20 | +147.100 | [+146.200, +148.000] | 0.0001 | 4.360 [4.271, 4.449] |
| BP-fixed - BP | energy_wh | 20 | +14.603 | [+14.314, +14.971] | 0.0000 | 4.382 [4.254, 4.533] |
| BP-fixed - BP | wall_sec | 20 | +716.842 | [+705.734, +729.531] | 0.0000 | 4.380 [4.265, 4.502] |
| MF-Seq Protocol B - BP | test_accuracy | 20 | +0.017 | [-0.138, +0.173] | 0.9851 | n/a |
| MF-Seq Protocol B - BP | passes | 20 | +0.100 | [-0.800, +1.000] | 1.0000 | 1.004 [0.984, 1.025] |
| MF-Seq Protocol B - BP | energy_wh | 20 | -0.633 | [-0.754, -0.508] | 0.0000 | 0.857 [0.832, 0.883] |
| MF-Seq Protocol B - BP | wall_sec | 20 | -2.645 | [-8.160, +2.831] | 0.5706 | 0.990 [0.965, 1.016] |
| nom0 Protocol B - MF-Seq Protocol B | test_accuracy | 20 | +1.112 | [+0.920, +1.302] | 0.0001 | n/a |
| nom0 Protocol B - MF-Seq Protocol B | passes | 20 | +1.000 | [+1.000, +1.000] | 0.0000 | 1.023 [1.023, 1.023] |
| nom0 Protocol B - MF-Seq Protocol B | energy_wh | 20 | +0.099 | [+0.029, +0.172] | 0.0172 | 1.028 [1.009, 1.048] |
| nom0 Protocol B - MF-Seq Protocol B | wall_sec | 20 | +5.093 | [+3.500, +6.727] | 0.0000 | 1.024 [1.017, 1.032] |
| MF-Seq cap-300 - MF-Seq | test_accuracy | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | passes | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | energy_wh | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | wall_sec | 0 | n/a | [n/a, n/a] | n/a | n/a |

### cifar10 descriptive controls (raw CIs; not in Holm)

| control | metric | n | mean diff | 95% CI | raw p | ratio [95% CI] |
|---|---|---|---|---|---|---|
| BP-bestval - BP | test_accuracy | 20 | +0.173 | [+0.012, +0.334] | 0.0707 | n/a |
| BP-bestval - BP | passes | 20 | +0.000 | [+0.000, +0.000] | 1.0000 | 1.000 [1.000, 1.000] |
| BP-bestval - BP | energy_wh | 20 | +0.110 | [+0.035, +0.184] | 0.0192 | 1.022 [1.008, 1.036] |
| BP-bestval - BP | wall_sec | 20 | +3.084 | [-0.122, +6.093] | 0.0400 | 1.012 [1.000, 1.024] |
| BP-sched - BP | test_accuracy | 20 | +3.061 | [+2.883, +3.229] | 0.0000 | n/a |
| BP-sched - BP | passes | 20 | +47.050 | [+45.600, +48.450] | 0.0001 | 1.896 [1.844, 1.946] |
| BP-sched - BP | energy_wh | 20 | +4.804 | [+4.584, +5.055] | 0.0000 | 1.940 [1.873, 2.014] |
| BP-sched - BP | wall_sec | 20 | +236.983 | [+225.209, +250.860] | 0.0000 | 1.938 [1.868, 2.018] |
| BP-fixed - BP | test_accuracy | 20 | +3.119 | [+2.919, +3.305] | 0.0000 | n/a |
| BP-fixed - BP | passes | 20 | +158.050 | [+156.600, +159.450] | 0.0001 | 4.000 [3.891, 4.107] |
| BP-fixed - BP | energy_wh | 20 | +15.826 | [+15.477, +16.212] | 0.0000 | 4.088 [3.937, 4.244] |
| BP-fixed - BP | wall_sec | 20 | +781.544 | [+762.590, +803.125] | 0.0000 | 4.083 [3.929, 4.244] |
| MF-Seq Protocol B - BP | test_accuracy | 20 | -0.810 | [-1.054, -0.581] | 0.0000 | n/a |
| MF-Seq Protocol B - BP | passes | 20 | +3.050 | [+1.600, +4.450] | 0.0018 | 1.062 [1.033, 1.090] |
| MF-Seq Protocol B - BP | energy_wh | 20 | -0.476 | [-0.651, -0.315] | 0.0000 | 0.912 [0.883, 0.941] |
| MF-Seq Protocol B - BP | wall_sec | 20 | +10.432 | [+2.520, +17.945] | 0.0136 | 1.046 [1.015, 1.077] |
| nom0 Protocol B - MF-Seq Protocol B | test_accuracy | 20 | +0.819 | [+0.644, +0.999] | 0.0001 | n/a |
| nom0 Protocol B - MF-Seq Protocol B | passes | 20 | -2.000 | [-2.000, -2.000] | 0.0000 | 0.964 [0.964, 0.964] |
| nom0 Protocol B - MF-Seq Protocol B | energy_wh | 20 | -0.100 | [-0.151, -0.043] | 0.0017 | 0.979 [0.968, 0.991] |
| nom0 Protocol B - MF-Seq Protocol B | wall_sec | 20 | -7.609 | [-9.451, -5.773] | 0.0000 | 0.971 [0.965, 0.978] |
| MF-Seq cap-300 - MF-Seq | test_accuracy | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | passes | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | energy_wh | 0 | n/a | [n/a, n/a] | n/a | n/a |
| MF-Seq cap-300 - MF-Seq | wall_sec | 0 | n/a | [n/a, n/a] | n/a | n/a |

## Decision notes

H-W1a/H-W1c conjunction p-values are the conservative intersection-union maximum of their component accuracy tests. H-W1a and H-W1b are threshold-exclusive; a point estimate in (-2.0, -1.0) pp is intermediate. H-W1d and H-W1cap are controls with raw CIs, never Holm-corrected. No conclusion is drawn from an arm with fewer than 20 paired seeds.

## Secondary diagnostic tables (not in Holm or any verdict)

These are post-training observations from each run's own JSON. They never enter a decision rule. `secondary_eval_seconds` is measured after the NVML monitor stops; the full run duration is not used for the wall endpoint.

| arm | dataset | n | train acc | train loss | val acc | val loss | test acc | test loss | probe sec |
|---|---|---|---|---|---|---|---|---|---|
| BP | cifar10 | 20 | 86.10 | 0.4203 | 62.41 | 1.1992 | 62.47 | 1.2012 | 10.621 |
| BP | cifar100 | 20 | 71.97 | 1.0587 | 32.33 | 3.0986 | 32.90 | 3.0863 | 10.638 |
| BP-bestval | cifar10 | 20 | 83.72 | 0.4836 | 63.15 | 1.1598 | 62.64 | 1.1683 | 10.831 |
| BP-bestval | cifar100 | 20 | 66.95 | 1.2458 | 32.81 | 3.0062 | 32.88 | 2.9966 | 10.775 |
| BP-fixed | cifar10 | 20 | 97.66 | 0.1152 | 65.70 | 1.3124 | 65.59 | 1.3035 | 10.762 |
| BP-fixed | cifar100 | 20 | 98.50 | 0.1313 | 33.60 | 4.0240 | 34.20 | 4.0203 | 10.664 |
| BP-sched | cifar10 | 20 | 83.29 | 0.5093 | 65.25 | 1.0374 | 65.53 | 1.0333 | 10.816 |
| BP-sched | cifar100 | 20 | 75.23 | 1.0086 | 34.37 | 2.9390 | 34.85 | 2.9306 | 10.696 |
| Seq-Global | cifar10 | 20 | 86.12 | 0.4071 | 65.46 | 1.1227 | 65.20 | 1.1200 | 10.892 |
| Seq-Global | cifar100 | 20 | 76.02 | 0.9129 | 37.80 | 2.6928 | 38.84 | 2.6692 | 10.910 |
| MF-Joint | cifar10 | 20 | 83.05 | 0.4884 | 62.14 | 1.2285 | 62.28 | 1.2310 | 10.807 |
| MF-Joint | cifar100 | 20 | 76.73 | 0.9114 | 34.03 | 2.9543 | 34.76 | 2.9405 | 10.893 |
| MF-Seq-nom0 | cifar10 | 20 | 88.62 | 0.3347 | 66.48 | 1.1676 | 66.44 | 1.1655 | 10.759 |
| MF-Seq-nom0 | cifar100 | 20 | 92.25 | 0.2920 | 37.20 | 3.4911 | 37.94 | 3.4592 | 10.703 |
| nom0 Protocol B | cifar10 | 20 | 77.67 | 0.6428 | 62.88 | 1.1236 | 62.48 | 1.1305 | 10.547 |
| nom0 Protocol B | cifar100 | 20 | 71.16 | 1.0786 | 33.20 | 3.0152 | 34.03 | 2.9967 | 10.581 |
| MF-Seq | cifar10 | 20 | 88.87 | 0.3291 | 66.51 | 1.1620 | 66.37 | 1.1673 | 10.743 |
| MF-Seq | cifar100 | 20 | 92.29 | 0.2910 | 37.05 | 3.4917 | 37.94 | 3.4534 | 10.785 |
| MF-Seq Protocol B | cifar10 | 20 | 73.26 | 0.7612 | 61.79 | 1.1234 | 61.66 | 1.1248 | 10.478 |
| MF-Seq Protocol B | cifar100 | 20 | 58.67 | 1.5460 | 32.15 | 2.9107 | 32.92 | 2.9010 | 10.590 |
| MF-Simul | cifar10 | 20 | 79.83 | 0.5763 | 61.84 | 1.1801 | 61.61 | 1.1853 | 10.841 |
| MF-Simul | cifar100 | 20 | 77.12 | 0.8347 | 31.89 | 3.3610 | 32.63 | 3.3503 | 10.730 |

### Linear probes and effective rank

| arm | dataset | layer | n | probe acc % | effective rank |
|---|---|---|---|---|---|
| BP | cifar10 | a_1 | 20 | 41.74 | 1021.397 |
| BP | cifar10 | a_2 | 20 | 49.64 | 1251.286 |
| BP | cifar10 | a_3 | 20 | 52.47 | 1202.274 |
| BP | cifar100 | a_1 | 20 | 15.93 | 813.492 |
| BP | cifar100 | a_2 | 20 | 20.72 | 1045.409 |
| BP | cifar100 | a_3 | 20 | 22.49 | 1081.515 |
| BP-bestval | cifar10 | a_1 | 20 | 41.47 | 1025.648 |
| BP-bestval | cifar10 | a_2 | 20 | 49.52 | 1246.066 |
| BP-bestval | cifar10 | a_3 | 20 | 52.68 | 1183.633 |
| BP-bestval | cifar100 | a_1 | 20 | 15.89 | 820.916 |
| BP-bestval | cifar100 | a_2 | 20 | 20.64 | 1037.665 |
| BP-bestval | cifar100 | a_3 | 20 | 22.45 | 1057.614 |
| BP-fixed | cifar10 | a_1 | 20 | 42.64 | 998.292 |
| BP-fixed | cifar10 | a_2 | 20 | 50.11 | 1193.735 |
| BP-fixed | cifar10 | a_3 | 20 | 53.96 | 1004.950 |
| BP-fixed | cifar100 | a_1 | 20 | 16.55 | 726.214 |
| BP-fixed | cifar100 | a_2 | 20 | 20.65 | 946.386 |
| BP-fixed | cifar100 | a_3 | 20 | 22.29 | 1021.461 |
| BP-sched | cifar10 | a_1 | 20 | 41.37 | 1038.037 |
| BP-sched | cifar10 | a_2 | 20 | 49.53 | 1197.533 |
| BP-sched | cifar10 | a_3 | 20 | 53.82 | 948.976 |
| BP-sched | cifar100 | a_1 | 20 | 15.95 | 759.678 |
| BP-sched | cifar100 | a_2 | 20 | 20.21 | 900.132 |
| BP-sched | cifar100 | a_3 | 20 | 22.19 | 922.776 |
| Seq-Global | cifar10 | a_1 | 20 | 44.00 | 859.040 |
| Seq-Global | cifar10 | a_2 | 20 | 53.36 | 1367.671 |
| Seq-Global | cifar10 | a_3 | 20 | 56.35 | 1347.580 |
| Seq-Global | cifar100 | a_1 | 20 | 16.96 | 697.121 |
| Seq-Global | cifar100 | a_2 | 20 | 24.00 | 1211.987 |
| Seq-Global | cifar100 | a_3 | 20 | 26.14 | 1087.980 |
| MF-Joint | cifar10 | a_1 | 20 | 42.78 | 1140.588 |
| MF-Joint | cifar10 | a_2 | 20 | 50.89 | 1219.788 |
| MF-Joint | cifar10 | a_3 | 20 | 52.15 | 1452.765 |
| MF-Joint | cifar100 | a_1 | 20 | 16.40 | 930.792 |
| MF-Joint | cifar100 | a_2 | 20 | 22.16 | 1160.480 |
| MF-Joint | cifar100 | a_3 | 20 | 24.09 | 1140.502 |
| MF-Seq-nom0 | cifar10 | a_1 | 20 | 44.54 | 1070.738 |
| MF-Seq-nom0 | cifar10 | a_2 | 20 | 53.17 | 1405.512 |
| MF-Seq-nom0 | cifar10 | a_3 | 20 | 55.90 | 1510.012 |
| MF-Seq-nom0 | cifar100 | a_1 | 20 | 16.78 | 1010.510 |
| MF-Seq-nom0 | cifar100 | a_2 | 20 | 23.39 | 1429.777 |
| MF-Seq-nom0 | cifar100 | a_3 | 20 | 25.80 | 1488.716 |
| nom0 Protocol B | cifar10 | a_1 | 20 | 40.72 | 1142.387 |
| nom0 Protocol B | cifar10 | a_2 | 20 | 49.03 | 1238.063 |
| nom0 Protocol B | cifar10 | a_3 | 20 | 51.54 | 1459.303 |
| nom0 Protocol B | cifar100 | a_1 | 20 | 15.18 | 1079.573 |
| nom0 Protocol B | cifar100 | a_2 | 20 | 20.60 | 1270.960 |
| nom0 Protocol B | cifar100 | a_3 | 20 | 22.85 | 1440.252 |
| MF-Seq | cifar10 | a_1 | 20 | 43.99 | 1070.582 |
| MF-Seq | cifar10 | a_2 | 20 | 53.18 | 1406.404 |
| MF-Seq | cifar10 | a_3 | 20 | 56.18 | 1510.773 |
| MF-Seq | cifar100 | a_1 | 20 | 16.87 | 1009.509 |
| MF-Seq | cifar100 | a_2 | 20 | 23.65 | 1431.316 |
| MF-Seq | cifar100 | a_3 | 20 | 26.14 | 1489.120 |
| MF-Seq Protocol B | cifar10 | a_1 | 20 | 39.82 | 1147.086 |
| MF-Seq Protocol B | cifar10 | a_2 | 20 | 48.22 | 1197.207 |
| MF-Seq Protocol B | cifar10 | a_3 | 20 | 50.78 | 1416.814 |
| MF-Seq Protocol B | cifar100 | a_1 | 20 | 14.88 | 1077.997 |
| MF-Seq Protocol B | cifar100 | a_2 | 20 | 19.55 | 1206.112 |
| MF-Seq Protocol B | cifar100 | a_3 | 20 | 22.13 | 1400.338 |
| MF-Simul | cifar10 | a_1 | 20 | 42.31 | 737.592 |
| MF-Simul | cifar10 | a_2 | 20 | 53.37 | 968.690 |
| MF-Simul | cifar10 | a_3 | 20 | 52.58 | 1379.578 |
| MF-Simul | cifar100 | a_1 | 20 | 16.19 | 1031.245 |
| MF-Simul | cifar100 | a_2 | 20 | 22.98 | 988.485 |
| MF-Simul | cifar100 | a_3 | 20 | 23.46 | 1430.619 |

### Local-head accuracy

| arm | dataset | head | n | accuracy % |
|---|---|---|---|---|
| Seq-Global | cifar10 | M_0 | 20 | 39.46 |
| Seq-Global | cifar10 | M_1 | 20 | 52.23 |
| Seq-Global | cifar10 | M_2 | 20 | 65.36 |
| Seq-Global | cifar10 | M_3 | 20 | 65.79 |
| Seq-Global | cifar100 | M_0 | 20 | 16.81 |
| Seq-Global | cifar100 | M_1 | 20 | 25.78 |
| Seq-Global | cifar100 | M_2 | 20 | 37.03 |
| Seq-Global | cifar100 | M_3 | 20 | 39.20 |
| MF-Joint | cifar10 | M_0 | 20 | 38.76 |
| MF-Joint | cifar10 | M_1 | 20 | 59.01 |
| MF-Joint | cifar10 | M_2 | 20 | 63.32 |
| MF-Joint | cifar10 | M_3 | 20 | 63.01 |
| MF-Joint | cifar100 | M_0 | 20 | 16.85 |
| MF-Joint | cifar100 | M_1 | 20 | 32.83 |
| MF-Joint | cifar100 | M_2 | 20 | 35.39 |
| MF-Joint | cifar100 | M_3 | 20 | 35.28 |
| MF-Seq-nom0 | cifar10 | M_0 | 20 | 9.41 |
| MF-Seq-nom0 | cifar10 | M_1 | 20 | 61.69 |
| MF-Seq-nom0 | cifar10 | M_2 | 20 | 66.81 |
| MF-Seq-nom0 | cifar10 | M_3 | 20 | 67.06 |
| MF-Seq-nom0 | cifar100 | M_0 | 20 | 0.93 |
| MF-Seq-nom0 | cifar100 | M_1 | 20 | 34.05 |
| MF-Seq-nom0 | cifar100 | M_2 | 20 | 38.54 |
| MF-Seq-nom0 | cifar100 | M_3 | 20 | 38.42 |
| nom0 Protocol B | cifar10 | M_0 | 20 | 9.41 |
| nom0 Protocol B | cifar10 | M_1 | 20 | 55.51 |
| nom0 Protocol B | cifar10 | M_2 | 20 | 61.92 |
| nom0 Protocol B | cifar10 | M_3 | 20 | 63.27 |
| nom0 Protocol B | cifar100 | M_0 | 20 | 0.93 |
| nom0 Protocol B | cifar100 | M_1 | 20 | 27.16 |
| nom0 Protocol B | cifar100 | M_2 | 20 | 33.25 |
| nom0 Protocol B | cifar100 | M_3 | 20 | 34.50 |
| MF-Seq | cifar10 | M_0 | 20 | 38.76 |
| MF-Seq | cifar10 | M_1 | 20 | 61.91 |
| MF-Seq | cifar10 | M_2 | 20 | 66.53 |
| MF-Seq | cifar10 | M_3 | 20 | 66.91 |
| MF-Seq | cifar100 | M_0 | 20 | 15.03 |
| MF-Seq | cifar100 | M_1 | 20 | 34.02 |
| MF-Seq | cifar100 | M_2 | 20 | 38.34 |
| MF-Seq | cifar100 | M_3 | 20 | 38.30 |
| MF-Seq Protocol B | cifar10 | M_0 | 20 | 38.47 |
| MF-Seq Protocol B | cifar10 | M_1 | 20 | 54.59 |
| MF-Seq Protocol B | cifar10 | M_2 | 20 | 61.10 |
| MF-Seq Protocol B | cifar10 | M_3 | 20 | 62.42 |
| MF-Seq Protocol B | cifar100 | M_0 | 20 | 14.70 |
| MF-Seq Protocol B | cifar100 | M_1 | 20 | 25.81 |
| MF-Seq Protocol B | cifar100 | M_2 | 20 | 31.52 |
| MF-Seq Protocol B | cifar100 | M_3 | 20 | 33.29 |
| MF-Simul | cifar10 | M_0 | 20 | 38.50 |
| MF-Simul | cifar10 | M_1 | 20 | 59.13 |
| MF-Simul | cifar10 | M_2 | 20 | 62.91 |
| MF-Simul | cifar10 | M_3 | 20 | 62.46 |
| MF-Simul | cifar100 | M_0 | 20 | 14.92 |
| MF-Simul | cifar100 | M_1 | 20 | 30.75 |
| MF-Simul | cifar100 | M_2 | 20 | 33.84 |
| MF-Simul | cifar100 | M_3 | 20 | 32.98 |

### Stage epochs and stopping

| arm | dataset | stage | n | mean epochs | mean max | stopped early |
|---|---|---|---|---|---|---|
| BP | cifar10 | backprop | 20 | 52.95 | 100.0 | n/a |
| BP | cifar100 | backprop | 20 | 43.90 | 100.0 | n/a |
| BP-bestval | cifar10 | backprop | 20 | 52.95 | 100.0 | n/a |
| BP-bestval | cifar100 | backprop | 20 | 43.90 | 100.0 | n/a |
| BP-fixed | cifar10 | backprop | 20 | 211.00 | 211.0 | n/a |
| BP-fixed | cifar100 | backprop | 20 | 191.00 | 191.0 | n/a |
| BP-sched | cifar10 | backprop | 20 | 100.00 | 100.0 | n/a |
| BP-sched | cifar100 | backprop | 20 | 100.00 | 100.0 | n/a |
| Seq-Global | cifar10 | Layer_W1_M1 | 20 | 100.00 | 100.0 | n/a |
| Seq-Global | cifar10 | Layer_W2_M2 | 20 | 59.75 | 100.0 | n/a |
| Seq-Global | cifar10 | Layer_W3_M3 | 20 | 23.50 | 100.0 | n/a |
| Seq-Global | cifar10 | M0 | 20 | 50.90 | 100.0 | n/a |
| Seq-Global | cifar100 | Layer_W1_M1 | 20 | 100.00 | 100.0 | n/a |
| Seq-Global | cifar100 | Layer_W2_M2 | 20 | 62.80 | 100.0 | n/a |
| Seq-Global | cifar100 | Layer_W3_M3 | 20 | 30.55 | 100.0 | n/a |
| Seq-Global | cifar100 | M0 | 20 | 65.05 | 100.0 | n/a |
| MF-Joint | cifar10 | joint | 20 | 51.30 | 100.0 | n/a |
| MF-Joint | cifar100 | joint | 20 | 50.25 | 100.0 | n/a |
| MF-Seq-nom0 | cifar10 | Layer_W1_M1 | 20 | 99.30 | 100.0 | n/a |
| MF-Seq-nom0 | cifar10 | Layer_W2_M2 | 20 | 39.45 | 100.0 | n/a |
| MF-Seq-nom0 | cifar10 | Layer_W3_M3 | 20 | 21.40 | 100.0 | n/a |
| MF-Seq-nom0 | cifar100 | Layer_W1_M1 | 20 | 95.30 | 100.0 | n/a |
| MF-Seq-nom0 | cifar100 | Layer_W2_M2 | 20 | 30.05 | 100.0 | n/a |
| MF-Seq-nom0 | cifar100 | Layer_W3_M3 | 20 | 21.75 | 100.0 | n/a |
| nom0 Protocol B | cifar10 | Layer_W1_M1 | 20 | 18.00 | 18.0 | n/a |
| nom0 Protocol B | cifar10 | Layer_W2_M2 | 20 | 18.00 | 18.0 | n/a |
| nom0 Protocol B | cifar10 | Layer_W3_M3 | 20 | 18.00 | 18.0 | n/a |
| nom0 Protocol B | cifar100 | Layer_W1_M1 | 20 | 15.00 | 15.0 | n/a |
| nom0 Protocol B | cifar100 | Layer_W2_M2 | 20 | 15.00 | 15.0 | n/a |
| nom0 Protocol B | cifar100 | Layer_W3_M3 | 20 | 15.00 | 15.0 | n/a |
| MF-Seq | cifar10 | Layer_W1_M1 | 20 | 99.85 | 100.0 | n/a |
| MF-Seq | cifar10 | Layer_W2_M2 | 20 | 40.40 | 100.0 | n/a |
| MF-Seq | cifar10 | Layer_W3_M3 | 20 | 21.45 | 100.0 | n/a |
| MF-Seq | cifar10 | M0 | 20 | 49.50 | 100.0 | n/a |
| MF-Seq | cifar100 | Layer_W1_M1 | 20 | 95.05 | 100.0 | n/a |
| MF-Seq | cifar100 | Layer_W2_M2 | 20 | 30.35 | 100.0 | n/a |
| MF-Seq | cifar100 | Layer_W3_M3 | 20 | 21.60 | 100.0 | n/a |
| MF-Seq | cifar100 | M0 | 20 | 44.35 | 100.0 | n/a |
| MF-Seq Protocol B | cifar10 | Layer_W1_M1 | 20 | 14.00 | 14.0 | n/a |
| MF-Seq Protocol B | cifar10 | Layer_W2_M2 | 20 | 14.00 | 14.0 | n/a |
| MF-Seq Protocol B | cifar10 | Layer_W3_M3 | 20 | 14.00 | 14.0 | n/a |
| MF-Seq Protocol B | cifar10 | M0 | 20 | 14.00 | 14.0 | n/a |
| MF-Seq Protocol B | cifar100 | Layer_W1_M1 | 20 | 11.00 | 11.0 | n/a |
| MF-Seq Protocol B | cifar100 | Layer_W2_M2 | 20 | 11.00 | 11.0 | n/a |
| MF-Seq Protocol B | cifar100 | Layer_W3_M3 | 20 | 11.00 | 11.0 | n/a |
| MF-Seq Protocol B | cifar100 | M0 | 20 | 11.00 | 11.0 | n/a |
| MF-Simul | cifar10 | joint | 20 | 58.25 | 100.0 | n/a |
| MF-Simul | cifar100 | joint | 20 | 35.70 | 100.0 | n/a |

## Tuning disclosure and sources

Seq-Global and BP-sched each receive 50 lr/wd trials per dataset; BP-bestval and both Protocol-B arms inherit frozen values and add 0 trials. The six core arms keep their original budgets.

Passes come from each run JSON's `epochs_completed`; energy comes strictly from the exact `monitoring_csv_path` in that JSON. The shared append-only run log is never read. `wall_sec` is `training_duration_sec`, so instrumentation outside the NVML window cannot change the wall endpoint; `total_run_duration_sec` is retained as a descriptive column.
