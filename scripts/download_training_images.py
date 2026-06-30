from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

try:
    import requests
except ModuleNotFoundError:
    requests = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from PIL import Image, UnidentifiedImageError
except ModuleNotFoundError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass

OPENVERSE_IMAGES_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"

DEFAULT_USER_AGENT = (
    "TrabalhoDOIVImageCollector/0.1 "
    "(academic object detection dataset; contact: local project)"
)

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@dataclass
class SearchResult:
    provider: str
    class_name: str
    query: str
    title: str
    image_url: str
    page_url: str
    license_name: str
    license_url: str
    creator: str
    source: str


@dataclass
class DownloadRecord:
    class_name: str
    file_path: str
    sha256: str
    width: int
    height: int
    provider: str
    query: str
    title: str
    image_url: str
    page_url: str
    license_name: str
    license_url: str
    creator: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procura e baixa imagens brutas por classe para posterior anotacao "
            "em um dataset YOLO."
        )
    )
    parser.add_argument(
        "--queries",
        default="data/image_search_queries.yaml",
        help="Arquivo YAML com classes e termos de busca.",
    )
    parser.add_argument(
        "--output",
        default="datasets/raw/3d_printing_defects",
        help="Pasta onde as imagens serao salvas.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["openverse", "wikimedia"],
        default=["openverse", "wikimedia"],
        help="Fontes usadas para busca.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Baixa apenas classes especificas do YAML.",
    )
    parser.add_argument("--max-per-class", type=int, default=30)
    parser.add_argument("--results-per-query", type=int, default=25)
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--max-mb", type=float, default=8.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent usado nas APIs e downloads.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing = []
    if requests is None:
        missing.append("requests")
    if yaml is None:
        missing.append("PyYAML")
    if Image is None:
        missing.append("Pillow")

    if missing:
        raise SystemExit(
            "Dependencias ausentes: "
            + ", ".join(missing)
            + "\nInstale com: pip install -r requirements.txt"
        )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "item"


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_queries(path: Path, selected_classes: list[str] | None) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de buscas nao encontrado: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = data.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("O YAML deve conter um bloco 'classes'.")

    normalized: dict[str, list[str]] = {}
    for class_name, queries in classes.items():
        if selected_classes and class_name not in selected_classes:
            continue
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Classe sem termos de busca validos: {class_name}")
        normalized[str(class_name)] = [str(query) for query in queries]

    if selected_classes:
        missing = sorted(set(selected_classes) - set(normalized))
        if missing:
            raise ValueError(f"Classes nao encontradas no YAML: {', '.join(missing)}")

    return normalized


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Api-User-Agent": user_agent,
            "Accept": "application/json,image/*;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip",
        }
    )
    return session


def search_openverse(
    session: requests.Session,
    class_name: str,
    query: str,
    limit: int,
    timeout: float,
) -> list[SearchResult]:
    params = {"q": query, "page_size": min(limit, 50)}
    response = session.get(OPENVERSE_IMAGES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    results: list[SearchResult] = []
    for item in payload.get("results", []):
        image_url = item.get("url") or item.get("thumbnail")
        if not image_url:
            continue

        results.append(
            SearchResult(
                provider="openverse",
                class_name=class_name,
                query=query,
                title=clean_text(item.get("title")),
                image_url=image_url,
                page_url=item.get("foreign_landing_url") or "",
                license_name=clean_text(item.get("license")),
                license_url=item.get("license_url") or "",
                creator=clean_text(item.get("creator")),
                source=clean_text(item.get("source")),
            )
        )

    return results


def search_wikimedia(
    session: requests.Session,
    class_name: str,
    query: str,
    limit: int,
    timeout: float,
) -> list[SearchResult]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "maxlag": 5,
    }
    response = session.get(WIKIMEDIA_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    results: list[SearchResult] = []
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        image_info = (page.get("imageinfo") or [{}])[0]
        mime = image_info.get("mime", "")
        image_url = image_info.get("url", "")
        if not image_url or not mime.startswith("image/"):
            continue

        metadata = image_info.get("extmetadata", {})
        license_name = clean_text(
            metadata.get("LicenseShortName", {}).get("value")
            or metadata.get("License", {}).get("value")
        )
        license_url = clean_text(metadata.get("LicenseUrl", {}).get("value"))
        creator = clean_text(
            metadata.get("Artist", {}).get("value")
            or metadata.get("Credit", {}).get("value")
        )

        results.append(
            SearchResult(
                provider="wikimedia",
                class_name=class_name,
                query=query,
                title=clean_text(page.get("title")),
                image_url=image_url,
                page_url=image_info.get("descriptionurl") or "",
                license_name=license_name,
                license_url=license_url,
                creator=creator,
                source="Wikimedia Commons",
            )
        )

    return results


def iter_search_results(
    session: requests.Session,
    providers: Iterable[str],
    class_name: str,
    queries: list[str],
    results_per_query: int,
    timeout: float,
    delay: float,
) -> Iterable[SearchResult]:
    for query in queries:
        for provider in providers:
            print(f"[busca] {class_name} | {provider} | {query}")
            try:
                if provider == "openverse":
                    results = search_openverse(
                        session, class_name, query, results_per_query, timeout
                    )
                elif provider == "wikimedia":
                    results = search_wikimedia(
                        session, class_name, query, results_per_query, timeout
                    )
                else:
                    results = []
            except requests.RequestException as error:
                print(f"  falhou: {error}")
                results = []

            for result in results:
                yield result

            time.sleep(delay)


def read_existing_metadata(metadata_csv: Path) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hashes: set[str] = set()
    if not metadata_csv.exists():
        return urls, hashes

    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("image_url"):
                urls.add(row["image_url"])
            if row.get("sha256"):
                hashes.add(row["sha256"])

    return urls, hashes


def image_extension(image: Image.Image) -> str:
    if image.format == "PNG":
        return ".png"
    if image.format == "WEBP":
        return ".webp"
    return ".jpg"


def download_image(
    session: requests.Session,
    result: SearchResult,
    output_dir: Path,
    existing_hashes: set[str],
    min_width: int,
    min_height: int,
    max_mb: float,
    timeout: float,
) -> DownloadRecord | None:
    response = session.get(result.image_url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        print(f"  ignorada: content-type nao suportado ({content_type})")
        return None

    max_bytes = int(max_mb * 1024 * 1024)
    content = response.content
    if len(content) > max_bytes:
        print(f"  ignorada: arquivo maior que {max_mb:.1f} MB")
        return None

    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 in existing_hashes:
        print("  ignorada: duplicada pelo hash")
        return None

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except UnidentifiedImageError:
        print("  ignorada: PIL nao reconheceu a imagem")
        return None

    width, height = image.size
    if width < min_width or height < min_height:
        print(f"  ignorada: resolucao baixa ({width}x{height})")
        return None

    class_dir = output_dir / result.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    extension = image_extension(image)
    file_name = f"{slugify(result.class_name)}_{sha256[:12]}{extension}"
    file_path = class_dir / file_name

    if extension == ".jpg":
        image = image.convert("RGB")
        image.save(file_path, quality=95, optimize=True)
    else:
        image.save(file_path)

    existing_hashes.add(sha256)
    return DownloadRecord(
        class_name=result.class_name,
        file_path=str(file_path.as_posix()),
        sha256=sha256,
        width=width,
        height=height,
        provider=result.provider,
        query=result.query,
        title=result.title,
        image_url=result.image_url,
        page_url=result.page_url,
        license_name=result.license_name,
        license_url=result.license_url,
        creator=result.creator,
        source=result.source,
    )


def append_metadata(metadata_csv: Path, metadata_jsonl: Path, records: list[DownloadRecord]) -> None:
    if not records:
        return

    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    write_header = not metadata_csv.exists()

    with metadata_csv.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    with metadata_jsonl.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    queries_path = Path(args.queries)
    output_dir = Path(args.output)
    metadata_csv = output_dir / "metadata.csv"
    metadata_jsonl = output_dir / "metadata.jsonl"

    class_queries = load_queries(queries_path, args.classes)
    session = make_session(args.user_agent)
    existing_urls, existing_hashes = read_existing_metadata(metadata_csv)

    print(f"Saida: {output_dir}")
    print(f"Fontes: {', '.join(args.providers)}")
    print(f"Classes: {', '.join(class_queries)}")

    all_records: list[DownloadRecord] = []
    for class_name, queries in class_queries.items():
        downloaded_for_class = 0
        seen_candidate_urls: set[str] = set()

        for result in iter_search_results(
            session=session,
            providers=args.providers,
            class_name=class_name,
            queries=queries,
            results_per_query=args.results_per_query,
            timeout=args.timeout,
            delay=args.delay,
        ):
            if downloaded_for_class >= args.max_per_class:
                break
            if result.image_url in existing_urls or result.image_url in seen_candidate_urls:
                continue
            seen_candidate_urls.add(result.image_url)

            if args.dry_run:
                print(f"  candidato: {result.title or result.image_url}")
                downloaded_for_class += 1
                continue

            try:
                record = download_image(
                    session=session,
                    result=result,
                    output_dir=output_dir,
                    existing_hashes=existing_hashes,
                    min_width=args.min_width,
                    min_height=args.min_height,
                    max_mb=args.max_mb,
                    timeout=args.timeout,
                )
            except requests.RequestException as error:
                print(f"  download falhou: {error}")
                record = None

            if record is None:
                continue

            existing_urls.add(result.image_url)
            all_records.append(record)
            downloaded_for_class += 1
            print(f"  salva: {record.file_path} ({record.width}x{record.height})")
            time.sleep(args.delay)

        print(f"[classe] {class_name}: {downloaded_for_class} novas imagens")

    append_metadata(metadata_csv, metadata_jsonl, all_records)

    print("\nColeta finalizada.")
    print(f"Novas imagens: {len(all_records)}")
    if not args.dry_run:
        print(f"Metadados: {metadata_csv}")
        print("Revise as imagens e anote bounding boxes antes do treino YOLO.")


if __name__ == "__main__":
    main()
