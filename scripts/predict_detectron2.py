from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera predicoes Detectron2 em imagens locais."
    )
    parser.add_argument("--dataset-root", default="datasets/Find3dprint.detectron2")
    parser.add_argument("--weights", default="runs/detectron2/find3dprint/model_final.pth")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", default=None, help="Imagem ou pasta. Se omitido, usa --split.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output", default="runs/detectron2/predictions")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_detectron2():
    try:
        import cv2
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Dependencia ausente para Detectron2.\n"
            "Instale no Ubuntu com:\n"
            "  pip install -r requirements-detectron2.txt\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            f"Erro original: {error}"
        ) from error

    return {
        "cv2": cv2,
        "torch": torch,
        "model_zoo": model_zoo,
        "get_cfg": get_cfg,
        "DatasetCatalog": DatasetCatalog,
        "MetadataCatalog": MetadataCatalog,
        "register_coco_instances": register_coco_instances,
        "DefaultPredictor": DefaultPredictor,
        "Visualizer": Visualizer,
    }


def annotation_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "annotations" / f"{split}.json"


def image_root(dataset_root: Path, split: str) -> Path:
    return dataset_root / "images" / split


def infer_num_classes(train_json: Path) -> int:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    return len(data.get("categories", []))


def register_dataset(modules: dict, dataset_root: Path, split: str, name: str) -> bool:
    json_file = annotation_path(dataset_root, split)
    images = image_root(dataset_root, split)
    if not json_file.exists() or not images.exists():
        return False

    DatasetCatalog = modules["DatasetCatalog"]
    MetadataCatalog = modules["MetadataCatalog"]
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        try:
            MetadataCatalog.remove(name)
        except KeyError:
            pass

    modules["register_coco_instances"](name, {}, str(json_file), str(images))
    return True


def build_cfg(modules: dict, args: argparse.Namespace):
    model_zoo = modules["model_zoo"]
    cfg = modules["get_cfg"]()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = infer_num_classes(
        annotation_path(Path(args.dataset_root), "train")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size

    torch = modules["torch"]
    if args.device == "auto":
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.MODEL.DEVICE = args.device

    return cfg


def images_from_source(source: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Fonte nao encontrada: {source}")


def images_from_split(dataset_root: Path, split: str, num_samples: int, seed: int) -> list[Path]:
    data = json.loads(annotation_path(dataset_root, split).read_text(encoding="utf-8"))
    root = image_root(dataset_root, split)
    images = [root / Path(image["file_name"]).name for image in data.get("images", [])]
    images = [image for image in images if image.exists()]
    random.Random(seed).shuffle(images)
    return images[:num_samples]


def main() -> None:
    args = parse_args()
    modules = require_detectron2()
    dataset_root = Path(args.dataset_root)
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Pesos nao encontrados: {weights}")

    dataset_name = f"find3dprint_{args.split}"
    if not register_dataset(modules, dataset_root, args.split, dataset_name):
        raise SystemExit(f"Split {args.split} nao encontrado em {dataset_root}")

    cfg = build_cfg(modules, args)
    predictor = modules["DefaultPredictor"](cfg)
    metadata = modules["MetadataCatalog"].get(dataset_name)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        images_from_source(args.source)
        if args.source
        else images_from_split(dataset_root, args.split, args.num_samples, args.seed)
    )

    cv2 = modules["cv2"]
    Visualizer = modules["Visualizer"]

    print(f"Gerando predicoes em {len(image_paths)} imagens.")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Imagem ignorada: {image_path}")
            continue

        outputs = predictor(image)
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        drawn = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = output_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(output_path), drawn.get_image()[:, :, ::-1])
        print(f"salvo: {output_path}")


if __name__ == "__main__":
    main()
