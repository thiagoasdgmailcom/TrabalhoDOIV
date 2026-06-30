from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara um COCO exportado pelo Roboflow para treino Detectron2."
    )
    parser.add_argument(
        "--input",
        default="datasets/Find3dprint.coco",
        help="Pasta do dataset COCO exportado pelo Roboflow.",
    )
    parser.add_argument(
        "--output",
        default="datasets/Find3dprint.detectron2",
        help="Pasta de saida preparada para Detectron2.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def discover_splits(input_dir: Path) -> list[tuple[str, str, Path]]:
    splits: list[tuple[str, str, Path]] = []
    for source_split, target_split in SPLIT_MAP.items():
        split_dir = input_dir / source_split
        annotation_file = split_dir / "_annotations.coco.json"
        if annotation_file.exists():
            splits.append((source_split, target_split, split_dir))
    return splits


def category_mapping(coco_by_split: dict[str, dict]) -> tuple[dict[int, int], list[dict]]:
    names_by_id: dict[int, str] = {}
    used_ids: set[int] = set()

    for coco in coco_by_split.values():
        for category in coco.get("categories", []):
            names_by_id[int(category["id"])] = str(category["name"])
        for annotation in coco.get("annotations", []):
            used_ids.add(int(annotation["category_id"]))

    if not used_ids:
        raise ValueError("Nenhuma categoria usada nas anotacoes.")

    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(used_ids), start=1)}
    categories = [
        {
            "id": new_id,
            "name": names_by_id.get(old_id, str(old_id)),
            "supercategory": "3d_print",
        }
        for old_id, new_id in mapping.items()
    ]
    return mapping, categories


def validate_image_exists(split_dir: Path, file_name: str) -> Path:
    image_path = split_dir / file_name
    if image_path.exists():
        return image_path

    fallback = split_dir / Path(file_name).name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Imagem nao encontrada: {split_dir / file_name}")


def remap_coco(coco: dict, mapping: dict[int, int], categories: list[dict]) -> dict:
    annotations = []
    for annotation in coco.get("annotations", []):
        old_category_id = int(annotation["category_id"])
        if old_category_id not in mapping:
            continue

        bbox = annotation.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue

        remapped = dict(annotation)
        remapped["category_id"] = mapping[old_category_id]
        annotations.append(remapped)

    prepared = dict(coco)
    prepared["categories"] = categories
    prepared["annotations"] = annotations
    return prepared


def copy_split_images(
    source_split_dir: Path,
    target_image_dir: Path,
    coco: dict,
    copy_images: bool,
) -> None:
    target_image_dir.mkdir(parents=True, exist_ok=True)
    for image in coco.get("images", []):
        source_image = validate_image_exists(source_split_dir, image["file_name"])
        target_image = target_image_dir / Path(image["file_name"]).name
        image["file_name"] = target_image.name
        if copy_images:
            shutil.copy2(source_image, target_image)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    annotations_dir = output_dir / "annotations"
    images_dir = output_dir / "images"

    splits = discover_splits(input_dir)
    if not splits:
        raise SystemExit(f"Nenhum split COCO encontrado em: {input_dir}")

    coco_by_split = {
        source_split: load_json(split_dir / "_annotations.coco.json")
        for source_split, _, split_dir in splits
    }
    mapping, categories = category_mapping(coco_by_split)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Entrada: {input_dir}")
    print(f"Saida:   {output_dir}")
    print("Classes:")
    for category in categories:
        print(f"- {category['id']}: {category['name']}")

    for source_split, target_split, split_dir in splits:
        coco = remap_coco(coco_by_split[source_split], mapping, categories)
        copy_split_images(
            source_split_dir=split_dir,
            target_image_dir=images_dir / target_split,
            coco=coco,
            copy_images=True,
        )

        output_json = annotations_dir / f"{target_split}.json"
        output_json.write_text(
            json.dumps(coco, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{source_split} -> {target_split}: "
            f"{len(coco.get('images', []))} imagens, "
            f"{len(coco.get('annotations', []))} anotacoes"
        )

    print("\nDataset Detectron2 pronto.")


if __name__ == "__main__":
    main()
