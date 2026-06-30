from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
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
        description="Converte dataset COCO exportado pelo Roboflow para YOLO."
    )
    parser.add_argument(
        "--input",
        default="datasets/Find3dprint.coco",
        help="Pasta do dataset COCO exportado pelo Roboflow.",
    )
    parser.add_argument(
        "--output",
        default="datasets/Find3dprint.yolo",
        help="Pasta de saida no formato YOLO.",
    )
    parser.add_argument(
        "--data-yaml",
        default="data/find3dprint.yaml",
        help="Arquivo YAML YOLO que sera criado.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copia as imagens para a pasta YOLO de saida.",
    )
    return parser.parse_args()


def load_coco(annotation_path: Path) -> dict:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Anotacao COCO nao encontrada: {annotation_path}")
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def find_splits(input_dir: Path) -> list[tuple[str, str, Path]]:
    splits: list[tuple[str, str, Path]] = []
    for source_split, yolo_split in SPLIT_MAP.items():
        split_dir = input_dir / source_split
        annotation_path = split_dir / "_annotations.coco.json"
        if annotation_path.exists():
            splits.append((source_split, yolo_split, split_dir))
    return splits


def build_category_mapping(coco_by_split: dict[str, dict]) -> tuple[dict[int, int], list[str]]:
    used_category_ids: set[int] = set()
    category_names: dict[int, str] = {}

    for coco in coco_by_split.values():
        for category in coco.get("categories", []):
            category_names[int(category["id"])] = str(category["name"])
        for annotation in coco.get("annotations", []):
            used_category_ids.add(int(annotation["category_id"]))

    if not used_category_ids:
        raise ValueError("Nenhuma categoria foi usada nas anotacoes COCO.")

    sorted_ids = sorted(used_category_ids)
    category_id_to_yolo_id = {
        category_id: yolo_id for yolo_id, category_id in enumerate(sorted_ids)
    }
    names = [category_names.get(category_id, str(category_id)) for category_id in sorted_ids]
    return category_id_to_yolo_id, names


def normalize_bbox(
    bbox: list[float],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = [float(value) for value in bbox]

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + clipped_width / 2) / image_width
    y_center = (y1 + clipped_height / 2) / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height
    return x_center, y_center, norm_width, norm_height


def image_source_path(split_dir: Path, file_name: str) -> Path:
    direct = split_dir / file_name
    if direct.exists():
        return direct

    matches = [
        path
        for path in split_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name == Path(file_name).name
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Imagem nao encontrada: {split_dir / file_name}")


def convert_split(
    coco: dict,
    split_dir: Path,
    output_dir: Path,
    yolo_split: str,
    category_id_to_yolo_id: dict[int, int],
    copy_images: bool,
) -> tuple[int, int]:
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    image_output_dir = output_dir / "images" / yolo_split
    label_output_dir = output_dir / "labels" / yolo_split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    converted_annotations = 0
    for image_id, image in images_by_id.items():
        file_name = image["file_name"]
        source_image = image_source_path(split_dir, file_name)
        target_image = image_output_dir / source_image.name
        if copy_images:
            shutil.copy2(source_image, target_image)

        width = float(image["width"])
        height = float(image["height"])
        label_lines: list[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id not in category_id_to_yolo_id:
                continue

            normalized = normalize_bbox(annotation["bbox"], width, height)
            if normalized is None:
                continue

            yolo_id = category_id_to_yolo_id[category_id]
            values = " ".join(f"{value:.6f}" for value in normalized)
            label_lines.append(f"{yolo_id} {values}")
            converted_annotations += 1

        label_file = label_output_dir / f"{source_image.stem}.txt"
        label_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return len(images_by_id), converted_annotations


def write_data_yaml(data_yaml: Path, output_dir: Path, names: list[str]) -> None:
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    relative_output = Path("..") / output_dir.as_posix()
    lines = [
        "# Dataset convertido de COCO/Roboflow para YOLO.",
        f"path: {relative_output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for index, name in enumerate(names):
        escaped_name = name.replace('"', '\\"')
        lines.append(f'  {index}: "{escaped_name}"')

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    data_yaml = Path(args.data_yaml)

    splits = find_splits(input_dir)
    if not splits:
        raise SystemExit(f"Nenhum split COCO encontrado em: {input_dir}")

    coco_by_split = {
        source_split: load_coco(split_dir / "_annotations.coco.json")
        for source_split, _, split_dir in splits
    }
    category_id_to_yolo_id, names = build_category_mapping(coco_by_split)

    print(f"Entrada COCO: {input_dir}")
    print(f"Saida YOLO:   {output_dir}")
    print(f"Classes:      {names}")

    for source_split, yolo_split, split_dir in splits:
        image_count, annotation_count = convert_split(
            coco=coco_by_split[source_split],
            split_dir=split_dir,
            output_dir=output_dir,
            yolo_split=yolo_split,
            category_id_to_yolo_id=category_id_to_yolo_id,
            copy_images=args.copy_images,
        )
        print(
            f"{source_split} -> {yolo_split}: "
            f"{image_count} imagens, {annotation_count} anotacoes"
        )

    write_data_yaml(data_yaml, output_dir, names)
    print(f"YAML criado: {data_yaml}")
    print("\nAgora treine com:")
    print(f"python scripts/train.py --data {data_yaml.as_posix()} --epochs 80 --imgsz 640 --batch 8")


if __name__ == "__main__":
    main()
