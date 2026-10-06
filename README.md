# Small-Object Detection in Drone Imagery with Slicing and Efficient Detectors

Training-free **Density-Adaptive Slicing (DAS)** for small-object detection on VisDrone: budgeted tiles guided by a coarse detection prior and image energy, compared with full-image inference and SAHI on CPU.

**Authors:** Abu Bakar Abdullah, Abdul Mohsin  
**Code:** [this repository](https://github.com/abubakarabdullah014/Small-Object-Detection-in-Drone-Imagery-with-Slicing-and-Efficient-Detectors)

## What is in this repo

| Included | Not included (download locally) |
|----------|----------------------------------|
| `src/` experiment + DAS code | VisDrone images (`data/raw/`) |
| `configs/` frozen DAS / SAHI settings | Model weights `.pt` (`models/`) |
| `requirements.txt` | Detection dump JSONs |
| `data/tuning/` image id list for the 200-image tune set | Paper PDF / Overleaf files |
| Small result summaries under `results/` | |

## Setup

```bash
pip install -r requirements.txt
```

CPU-only is fine (no NVIDIA GPU required).

## 1. Dataset

```bash
python src/convert_visdrone.py --splits val test-dev train
```

Downloads VisDrone2019-DET, converts to COCO JSON (drops ignored / others), and writes the fixed 200-image tuning subset (`seed=42`).

## 2. Models

Public VisDrone fine-tuned YOLO checkpoints via Hugging Face / Ultralytics:

```bash
python src/zoo.py --download yolov8n yolov8s yolov11n yolov11s
```

## 3. Experiments

```bash
# Full-image baseline
python src/run_experiments.py --mode baseline --models yolov8n --split val

# SAHI
python src/run_experiments.py --mode sahi --models yolov8n --split val --quick

# DAS (uses configs/das_frozen.yaml for v2 settings)
python src/run_experiments.py --mode das --models yolov8n --split val

# Hyperparameter tune on train subset only
python src/run_experiments.py --mode tune --models yolov8n --split tune
```

Headline full-val numbers (YOLOv8n, 548 images, CPU) are summarized in `results/RESULTS.md` and `results/das_v2/RESULTS_v2.md`.

## License / data

- Code in this repository: for research use with the paper.
- VisDrone: follow the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) terms.
- Third-party models (Ultralytics YOLO, etc.): follow their licenses.

## Citation

If you use this code, please cite the accompanying manuscript:

> Abu Bakar Abdullah and Abdul Mohsin, *Small-Object Detection in Drone Imagery with Slicing and Efficient Detectors*.
