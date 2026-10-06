"""
Model zoo: download / load VisDrone-finetuned YOLO and COCO D-FINE via SAHI.
Optionally export YOLO to OpenVINO IR for consistent CPU latency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (
    CROSS_DATASET_MODELS,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    IN_DOMAIN_MODELS,
    MODELS_DIR,
    YOLO_VISDRONE_NAMES,
)


def download_yolo_weights(name: str) -> Path:
    """Download best.pt for an in-domain YOLO model. Returns local path."""
    if name not in IN_DOMAIN_MODELS:
        raise KeyError(f"Unknown in-domain model: {name}")
    meta = IN_DOMAIN_MODELS[name]
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=meta["repo"],
        filename=meta["filename"],
        local_dir=str(MODELS_DIR / name),
    )
    return Path(path)


def export_yolo_openvino(name: str, imgsz: int = 640) -> Path:
    """Export YOLO .pt to OpenVINO IR. Returns path to .xml or directory."""
    from ultralytics import YOLO

    pt = download_yolo_weights(name)
    out_dir = MODELS_DIR / name / "openvino"
    # Ultralytics creates {stem}_openvino_model/
    model = YOLO(str(pt))
    exported = model.export(format="openvino", imgsz=imgsz, half=False)
    exported_path = Path(exported)
    print(f"[export] {name} -> {exported_path}")
    return exported_path


def load_sahi_model(
    name: str,
    conf: float = DEFAULT_CONF,
    device: str = DEFAULT_DEVICE,
    use_openvino: bool = False,
):
    """
    Load a SAHI AutoDetectionModel.
    name: key in IN_DOMAIN_MODELS or CROSS_DATASET_MODELS
    """
    from sahi import AutoDetectionModel

    if name in IN_DOMAIN_MODELS:
        if use_openvino:
            ov_path = MODELS_DIR / name
            # Prefer existing openvino export
            candidates = list(ov_path.rglob("*_openvino_model"))
            if not candidates:
                candidates = list(ov_path.rglob("*.xml"))
            if candidates:
                model_path = str(candidates[0])
            else:
                model_path = str(export_yolo_openvino(name))
        else:
            model_path = str(download_yolo_weights(name))
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            device=device,
        )
        return detection_model

    if name in CROSS_DATASET_MODELS:
        meta = CROSS_DATASET_MODELS[name]
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=meta["repo"],
            confidence_threshold=conf,
            device=device,
        )
        return detection_model

    raise KeyError(f"Unknown model: {name}")


def sahi_prediction_to_dicts(
    result,
    image_id: int,
    category_offset: int = 1,
    name_to_id: Optional[Dict[str, int]] = None,
    filter_coco_for_visdrone: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert SAHI PredictionResult object predictions to COCO-style dicts.
    YOLO VisDrone models use 0-indexed classes; COCO GT uses 1-10 category_id.
    D-FINE uses COCO names; we map back to VisDrone categories when possible.
    """
    from src import VISDRONE_CATEGORIES, VISDRONE_TO_COCO

    # Build reverse COCO-name -> VisDrone category ids
    coco_to_vd: Dict[str, List[int]] = {}
    for vd_id, coco_name in VISDRONE_TO_COCO.items():
        if coco_name is None:
            continue
        coco_to_vd.setdefault(coco_name, []).append(vd_id)

    out: List[Dict[str, Any]] = []
    for pred in result.object_prediction_list:
        score = float(pred.score.value)
        x1 = float(pred.bbox.minx)
        y1 = float(pred.bbox.miny)
        x2 = float(pred.bbox.maxx)
        y2 = float(pred.bbox.maxy)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue

        if filter_coco_for_visdrone:
            # D-FINE / COCO model
            cname = pred.category.name if pred.category.name else ""
            if cname not in coco_to_vd:
                continue
            # Prefer first mapped VisDrone id (e.g. person -> pedestrian)
            cat_id = coco_to_vd[cname][0]
        else:
            # YOLO VisDrone: category.id is 0-indexed
            cat_id = int(pred.category.id) + category_offset

        out.append(
            {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": score,
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return out


def predict_full_image(
    detection_model,
    image_path: str,
    conf: Optional[float] = None,
):
    """Single full-image prediction via SAHI (no slicing)."""
    from sahi.predict import get_prediction

    if conf is not None:
        detection_model.confidence_threshold = conf
    return get_prediction(image_path, detection_model)


def predict_sahi_sliced(
    detection_model,
    image_path: str,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    conf: Optional[float] = None,
):
    """Uniform SAHI sliced prediction."""
    from sahi.predict import get_sliced_prediction

    if conf is not None:
        detection_model.confidence_threshold = conf
    return get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_pred,
        verbose=0,
    )


def export_dfine_onnx_hint(name: str = "dfine_s") -> str:
    """
    D-FINE is loaded via Hugging Face Transformers through SAHI at inference time.
    For a standalone OpenVINO path (optional latency study), clone Peterande/D-FINE and run:

      python tools/deployment/export_onnx.py --check \\
        -c configs/dfine/dfine_hgnetv2_s_coco.yml -r model.pth --simplify
      # then:
      # ov.convert_model(onnx, input=[(\"images\",[1,3,640,640]),(\"orig_target_sizes\",[1,2])])

    This helper only documents the path; Transformers+SAHI is the default in this repo.
    """
    return (
        f"See Peterande/D-FINE export for {name}; "
        "default runtime uses AutoDetectionModel model_type=huggingface."
    )


def validate_model(name: str, image_paths: List[str], conf: float = 0.25) -> Dict:
    """Smoke-test a model on a few images."""
    model = load_sahi_model(name, conf=conf)
    cross = name in CROSS_DATASET_MODELS
    n_dets = 0
    for p in image_paths:
        res = predict_full_image(model, p)
        dets = sahi_prediction_to_dicts(
            res, image_id=0, filter_coco_for_visdrone=cross
        )
        n_dets += len(dets)
    return {"model": name, "images": len(image_paths), "total_dets": n_dets}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", nargs="+", default=list(IN_DOMAIN_MODELS.keys()))
    parser.add_argument("--export-openvino", action="store_true")
    parser.add_argument("--validate", type=str, default=None, help="image path")
    args = parser.parse_args()

    for name in args.download:
        if name in IN_DOMAIN_MODELS:
            p = download_yolo_weights(name)
            print(f"downloaded {name}: {p}")
            if args.export_openvino:
                export_yolo_openvino(name)
        else:
            print(f"skip download for {name} (HF transformers)")

    if args.validate:
        for name in args.download:
            print(validate_model(name, [args.validate]))
