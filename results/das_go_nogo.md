# DAS go / no-go (full VisDrone val, 548 images)

## Headline numbers (YOLOv8n, CPU)

| Method | AP_small | AP | mean ms | slices/img |
|--------|----------|------|---------|------------|
| Full image | 0.155 | 0.246 | ~352 | 0 |
| SAHI 512/0.2 + full | **0.177** | 0.234 | 5743 | ~4.0 |
| DAS N=4 | 0.131 | 0.216 | **2183** | ~4.0 |
| DAS N=6 | 0.128 | 0.212 | 3194 | ~6.0 |
| DAS N=8 | 0.126 | 0.209 | 3766 | ~7.9 |

## Decision: **benchmark-primary** (partial DAS latency story)

- **SAHI benchmark: GO** — clear AP_small gain over full image (+2.2 pts) at known latency cost.
- **DAS accuracy: NO-GO as currently configured** on full val — AP_small trails both full image and SAHI.
- **DAS latency: GO as efficiency knob** — ~2.6× faster than SAHI at N=4, with an explicit `N_max` budget.

Paper framing: lead with the modern CPU SAHI benchmark; present DAS as a latency-budgeted scheduler whose accuracy needs further tuning (density threshold / fusion), supported by ablations on the 50-image explore set.

Best tune hyperparameters (train subset): `tau=0.15`, `k=8.0`, `n_max=4`, fusion=`wbf`.
