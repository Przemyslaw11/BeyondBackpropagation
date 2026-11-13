# G2 summary

Units: 1200 sweep, 100 bridge; non-positive or missing net energy: 0.

| hypothesis | dataset | n | estimate | ci_low | ci_high | p_raw | complete | note | sign | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-G2a | cifar10 | 10 | 0.05218 | 0.05117 | 0.05306 | 1.94e-15 | True | slope of log a (MF-Simul) on log(batch x width), per repeat | nan | nan | consistent |
| H-G2b | cifar10 | 10 | 0.3742 | 0.3688 | 0.3788 | 1.17e-16 | True | log a (MF-Simul) at batch 8192, width 8000 | nan | nan | consistent |
| H-G2d | cifar10 | 20 | -1.083 | -1.302 | -0.8485 | 0.9999 | True | BP-MFinit minus BP test accuracy (pp); TOST +-0.5 | nan | nan | not consistent |
| H-G2c | cifar10 | 10 | 0.07911 | 0.07234 | 0.08767 | nan | True | mf_simul: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar10 | 10 | 0.9175 | 0.8982 | 0.9354 | nan | True | mf_stage1: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar10 | 10 | 1.119 | 1.084 | 1.145 | nan | True | mf_stage2: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar10 | 10 | 0.94 | 0.9124 | 0.9619 | nan | True | mf_stage3: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2a | cifar100 | 10 | 0.05391 | 0.05244 | 0.05505 | 3.122e-14 | True | slope of log a (MF-Simul) on log(batch x width), per repeat | nan | 9.365e-14 | supported |
| H-G2b | cifar100 | 10 | 0.368 | 0.3589 | 0.3775 | 3.496e-14 | True | log a (MF-Simul) at batch 8192, width 8000 | nan | 9.365e-14 | supported |
| H-G2d | cifar100 | 20 | -0.8075 | -0.963 | -0.6355 | 0.9991 | True | BP-MFinit minus BP test accuracy (pp); TOST +-0.5 | nan | 0.9991 | not supported |
| H-G2c | cifar100 | 10 | 0.02855 | 0.004733 | 0.04539 | nan | True | mf_simul: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar100 | 10 | 0.8948 | 0.8638 | 0.9242 | nan | True | mf_stage1: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar100 | 10 | 1.118 | 1.088 | 1.146 | nan | True | mf_stage2: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
| H-G2c | cifar100 | 10 | 0.9378 | 0.9218 | 0.9533 | nan | True | mf_stage3: host-bridge log a, descriptive | MF cheaper | nan | descriptive |
