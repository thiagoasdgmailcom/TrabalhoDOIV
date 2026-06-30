from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida ou testa um modelo YOLO treinado."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/yolov8n_3d_print_defects/weights/best.pt",
        help="Caminho para os pesos treinados.",
    )
    parser.add_argument("--data", default="data/3d_printing_defects.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    val_kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": args.name or f"val_{args.split}",
        "plots": True,
        "save_json": False,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)
    save_dir = Path(metrics.save_dir)

    print("\nMetricas principais")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Resultados salvos em: {save_dir}")


if __name__ == "__main__":
    main()
