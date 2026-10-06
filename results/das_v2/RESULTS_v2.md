# Improved DAS v2 — full val results (real runs)

## Hardware
Intel i7-1185G7 CPU, YOLOv8n VisDrone-finetuned, 548 val images.

## Headline (v2: fixed 512 slice, NMS fusion, coarse floor, tau=0.12)

| Method | AP_small | AP | ms |
|--------|----------|------|-----|
| Full image | 0.155 | 0.246 | 328 |
| SAHI 512/0.2+full (prior run) | 0.177 | 0.234 | 5743 |
| **DAS N=4 v2** | **0.186** | **0.256** | **2314** |
| **DAS N=6 v2** | **0.189** | **0.257** | **3144** |

## vs old DAS v1
| | AP_small | ms |
|--|----------|-----|
| DAS N=4 v1 | 0.131 | 2183 |
| DAS N=4 v2 | 0.186 | 2314 |

## Decision: **GO** for DAS contribution
DAS N=4 beats SAHI on AP_small (+0.009) at ~2.5× lower latency.
