from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ModuleNotFoundError:
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Augmentation:
    name: str
    hflip: bool = False
    vflip: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    color: float = 1.0
    sharpness: float = 1.0
    blur: bool = False


AUGMENTATIONS = [
    Augmentation("hflip", hflip=True),
    Augmentation("brightness_up", brightness=1.18),
    Augmentation("brightness_down", brightness=0.82),
    Augmentation("contrast_up", contrast=1.22),
    Augmentation("contrast_down", contrast=0.82),
    Augmentation("color_up", color=1.25),
    Augmentation("sharpness_up", sharpness=1.6),
    Augmentation("hflip_brightness", hflip=True, brightness=1.12, contrast=1.08),
    Augmentation("hflip_contrast", hflip=True, contrast=1.18, color=1.08),
    Augmentation("vflip", vflip=True),
    Augmentation("soft_blur", blur=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aumenta um dataset YOLO de deteccao com transformacoes simples."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/Find3dprint.yolo",
        help="Raiz do dataset YOLO com pastas images/ e labels/.",
    )
    parser.add_argument(
        "--train-copies",
        type=int,
        default=3,
        help="Quantidade de imagens aumentadas geradas por imagem de treino.",
    )
    parser.add_argument(
        "--val-copies",
        type=int,
        default=0,
        help="Quantidade de imagens aumentadas geradas por imagem de validacao.",
    )
    parser.add_argument(
        "--test-copies",
        type=int,
        default=4,
        help="Quantidade de imagens aumentadas geradas por imagem de teste.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Tambem aumenta imagens que ja possuem '__aug_' no nome.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    if Image is None:
        raise SystemExit(
            "Dependencia ausente: Pillow\n"
            "Instale com: pip install -r requirements.txt"
        )


def collect_images(image_dir: Path, include_augmented: bool) -> list[Path]:
    if not image_dir.exists():
        return []

    images = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not include_augmented and "__aug_" in path.stem:
            continue
        images.append(path)
    return images


def read_yolo_labels(label_path: Path) -> list[list[str]]:
    if not label_path.exists():
        return []

    labels: list[list[str]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label invalido em {label_path}: {raw_line}")
        labels.append(parts)

    return labels


def transform_labels(labels: list[list[str]], augmentation: Augmentation) -> list[str]:
    transformed = []
    for parts in labels:
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        if augmentation.hflip:
            x_center = 1.0 - x_center
        if augmentation.vflip:
            y_center = 1.0 - y_center

        transformed.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return transformed


def apply_augmentation(image: Image.Image, augmentation: Augmentation) -> Image.Image:
    augmented = image.convert("RGB")

    if augmentation.hflip:
        augmented = ImageOps.mirror(augmented)
    if augmentation.vflip:
        augmented = ImageOps.flip(augmented)
    if augmentation.brightness != 1.0:
        augmented = ImageEnhance.Brightness(augmented).enhance(augmentation.brightness)
    if augmentation.contrast != 1.0:
        augmented = ImageEnhance.Contrast(augmented).enhance(augmentation.contrast)
    if augmentation.color != 1.0:
        augmented = ImageEnhance.Color(augmented).enhance(augmentation.color)
    if augmentation.sharpness != 1.0:
        augmented = ImageEnhance.Sharpness(augmented).enhance(augmentation.sharpness)
    if augmentation.blur:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.65))

    return augmented


def split_copy_count(split: str, args: argparse.Namespace) -> int:
    if split == "train":
        return args.train_copies
    if split == "val":
        return args.val_copies
    if split == "test":
        return args.test_copies
    return 0


def augment_split(dataset: Path, split: str, copies: int, include_augmented: bool) -> tuple[int, int, int]:
    if copies <= 0:
        return 0, 0, 0

    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    images = collect_images(image_dir, include_augmented)
    created = 0
    skipped = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path)
        selected_augmentations = random.sample(
            AUGMENTATIONS,
            k=min(copies, len(AUGMENTATIONS)),
        )

        with Image.open(image_path) as image:
            for index, augmentation in enumerate(selected_augmentations, start=1):
                output_stem = f"{image_path.stem}__aug_{augmentation.name}_{index}"
                output_image = image_dir / f"{output_stem}.jpg"
                output_label = label_dir / f"{output_stem}.txt"

                if output_image.exists() or output_label.exists():
                    skipped += 1
                    continue

                augmented_image = apply_augmentation(image, augmentation)
                augmented_labels = transform_labels(labels, augmentation)

                augmented_image.save(output_image, quality=95, optimize=True)
                output_label.write_text(
                    "\n".join(augmented_labels)
                    + ("\n" if augmented_labels else ""),
                    encoding="utf-8",
                )
                created += 1

    return len(images), created, skipped


def count_images(dataset: Path, split: str) -> int:
    return len(collect_images(dataset / "images" / split, include_augmented=True))


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    random.seed(args.seed)

    dataset = Path(args.dataset)
    if not (dataset / "images").exists() or not (dataset / "labels").exists():
        raise SystemExit(f"Dataset YOLO invalido: {dataset}")

    print(f"Dataset: {dataset}")
    for split in ("train", "val", "test"):
        copies = split_copy_count(split, args)
        before = count_images(dataset, split)
        source_count, created, skipped = augment_split(
            dataset=dataset,
            split=split,
            copies=copies,
            include_augmented=args.include_augmented,
        )
        after = count_images(dataset, split)

        print(
            f"{split}: antes={before}, fontes={source_count}, "
            f"criadas={created}, puladas={skipped}, depois={after}"
        )

    print("\nAumento de dados finalizado.")


if __name__ == "__main__":
    main()
