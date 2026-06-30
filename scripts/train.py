from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina YOLOv8n para deteccao de falhas em impressao 3D."
    )
    parser.add_argument("--data", default="data/3d_printing_defects.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ex.: 0 para GPU, cpu para CPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolov8n_3d_print_defects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Desliga mosaic nos ultimos N epochs.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    if YOLO is None:
        raise SystemExit(
            "Dependencia ausente: ultralytics\n"
            "Instale com: pip install -r requirements.txt"
        )


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "pretrained": True,
        "close_mosaic": args.close_mosaic,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)

    print("\nTreino finalizado.")
    print(f"Resultados: {save_dir}")
    print(f"Melhor peso: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
