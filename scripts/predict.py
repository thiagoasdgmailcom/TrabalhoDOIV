from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa predicoes com YOLO em imagem, pasta, video ou webcam."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/yolov8n_3d_print_defects/weights/best.pt",
        help="Caminho para os pesos treinados.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Imagem, pasta, video, URL ou indice da camera. Ex.: 0",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    predict_kwargs = {
        "source": parse_source(args.source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "show": args.show,
        "save": True,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    save_dir = Path(results[0].save_dir) if results else Path(args.project) / args.name

    print(f"Predicoes salvas em: {save_dir}")


if __name__ == "__main__":
    main()
