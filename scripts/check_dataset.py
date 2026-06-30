from __future__ import annotations

import argparse
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica a estrutura basica de um dataset YOLO."
    )
    parser.add_argument(
        "--data",
        default="data/3d_printing_defects.yaml",
        help="Caminho para o arquivo YAML do dataset.",
    )
    return parser.parse_args()


def load_dataset_config(data_path: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {data_path}")

    with data_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_dataset_path(data_path: Path, config: dict) -> Path:
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def collect_images(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(
        image
        for image in split_dir.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_label_file(label_file: Path, num_classes: int) -> list[str]:
    errors: list[str] = []
    if not label_file.exists():
        return [f"label ausente: {label_file}"]

    for line_number, raw_line in enumerate(
        label_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_file}:{line_number} deve ter 5 colunas")
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_file}:{line_number} contem valor invalido")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(f"{label_file}:{line_number} class_id fora do intervalo")
        if any(value < 0 or value > 1 for value in values):
            errors.append(f"{label_file}:{line_number} coordenada fora de [0, 1]")

    return errors


def main() -> None:
    data_path = Path(parse_args().data).resolve()
    config = load_dataset_config(data_path)
    dataset_root = resolve_dataset_path(data_path, config)
    names = config.get("names", {})
    num_classes = len(names)

    print(f"Dataset: {dataset_root}")
    print(f"Classes ({num_classes}): {names}")

    total_errors: list[str] = []
    for split in ("train", "val", "test"):
        image_rel = Path(config.get(split, f"images/{split}"))
        image_dir = dataset_root / image_rel
        label_dir = dataset_root / "labels" / split
        images = collect_images(image_dir)

        print(f"{split}: {len(images)} imagens em {image_dir}")

        if split in ("train", "val") and not images:
            total_errors.append(f"{split}: nenhuma imagem encontrada")

        for image in images:
            label_file = label_dir / f"{image.stem}.txt"
            total_errors.extend(validate_label_file(label_file, num_classes))

    if total_errors:
        print("\nProblemas encontrados:")
        for error in total_errors[:50]:
            print(f"- {error}")
        if len(total_errors) > 50:
            print(f"- ... mais {len(total_errors) - 50} problemas")
        raise SystemExit(1)

    print("\nEstrutura do dataset parece valida.")


if __name__ == "__main__":
    main()
