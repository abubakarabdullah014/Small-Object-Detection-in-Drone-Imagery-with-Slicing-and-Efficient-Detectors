# Experiment results summary

## Hardware
Intel i7-1185G7 (4C/8T), 16 GB RAM, Intel Iris Xe, CPU-only, 4 threads.

## Full-image baselines (val, 548)
| Model | AP | AP_small | ms |
|-------|------|----------|-----|
| YOLOv8n | 0.246 | 0.155 | 328 |
| YOLOv11n | 0.247 | 0.155 | 307 |
| YOLOv8s | 0.295 | 0.205 | 923 |
| YOLOv11s | 0.292 | 0.201 | 673 |

## Full-val slicing (YOLOv8n, 548) — DAS v2 improved
| Method | AP_small | AP | ms |
|--------|----------|------|-----|
| Full image | 0.155 | 0.246 | 328 |
| SAHI 512/0.2+full | 0.177 | 0.234 | 5743 |
| **DAS N=4 v2** | **0.186** | **0.256** | **2314** |
| **DAS N=6 v2** | **0.189** | **0.257** | **3144** |

DAS v2 changes: fixed 512 slices, NMS fusion, coarse-as-floor, tau=0.12.
Details: `results/das_v2/RESULTS_v2.md`
