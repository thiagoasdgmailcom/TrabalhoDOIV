from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina Faster R-CNN/Detectron2 em um dataset COCO local."
    )
    parser.add_argument("--dataset-root", default="datasets/Find3dprint.detectron2")
    parser.add_argument("--output", default="runs/detectron2/find3dprint")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=None, help="Pesos iniciais opcionais.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--base-lr", type=float, default=0.00025)
    parser.add_argument("--ims-per-batch", type=int, default=2)
    parser.add_argument("--roi-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-period", type=int, default=100)
    parser.add_argument("--checkpoint-period", type=int, default=250)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    return parser.parse_args()


def require_detectron2():
    try:
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data import build_detection_test_loader
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Dependencia ausente para Detectron2.\n"
            "Instale no Ubuntu com:\n"
            "  pip install -r requirements-detectron2.txt\n"
            "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n"
            f"Erro original: {error}"
        ) from error

    return {
        "torch": torch,
        "model_zoo": model_zoo,
        "get_cfg": get_cfg,
        "DatasetCatalog": DatasetCatalog,
        "MetadataCatalog": MetadataCatalog,
        "build_detection_test_loader": build_detection_test_loader,
        "register_coco_instances": register_coco_instances,
        "DefaultTrainer": DefaultTrainer,
        "COCOEvaluator": COCOEvaluator,
        "inference_on_dataset": inference_on_dataset,
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


def build_cfg(modules: dict, args: argparse.Namespace, train_name: str, val_name: str):
    model_zoo = modules["model_zoo"]
    cfg = modules["get_cfg"]()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))

    dataset_root = Path(args.dataset_root)
    num_classes = infer_num_classes(annotation_path(dataset_root, "train"))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = args.weights or model_zoo.get_checkpoint_url(args.config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size,)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    cfg.OUTPUT_DIR = args.output

    torch = modules["torch"]
    if args.device == "auto":
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.MODEL.DEVICE = args.device

    return cfg


def make_evaluator(modules: dict, dataset_name: str, output_dir: Path):
    COCOEvaluator = modules["COCOEvaluator"]
    try:
        return COCOEvaluator(dataset_name, output_dir=str(output_dir))
    except TypeError:
        return COCOEvaluator(dataset_name, None, False, output_dir=str(output_dir))


def main() -> None:
    args = parse_args()
    modules = require_detectron2()
    dataset_root = Path(args.dataset_root)

    train_name = "find3dprint_train"
    val_name = "find3dprint_val"
    test_name = "find3dprint_test"

    if not register_dataset(modules, dataset_root, "train", train_name):
        raise SystemExit(f"Split train nao encontrado em {dataset_root}")
    if not register_dataset(modules, dataset_root, "val", val_name):
        raise SystemExit(f"Split val nao encontrado em {dataset_root}")
    has_test = register_dataset(modules, dataset_root, "test", test_name)

    cfg = build_cfg(modules, args, train_name, val_name)
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    print(f"Dataset: {dataset_root}")
    print(f"Saida:   {output_dir}")
    print(f"Modelo:  {args.config}")
    print(f"Device:  {cfg.MODEL.DEVICE}")
    print(f"Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

    trainer = modules["DefaultTrainer"](cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    cfg.MODEL.WEIGHTS = str(output_dir / "model_final.pth")
    predictor_model = trainer.model

    for split_name, dataset_name in [("val", val_name), ("test", test_name)]:
        if split_name == "test" and (not args.eval_test or not has_test):
            continue
        evaluator = make_evaluator(modules, dataset_name, output_dir / f"eval_{split_name}")
        loader = modules["build_detection_test_loader"](cfg, dataset_name)
        results = modules["inference_on_dataset"](predictor_model, loader, evaluator)
        print(f"Metricas {split_name}: {results}")

    print(f"\nTreino finalizado. Pesos: {output_dir / 'model_final.pth'}")


if __name__ == "__main__":
    main()
