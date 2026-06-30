from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

try:
    import requests
except ModuleNotFoundError:
    requests = None

try:
    from PIL import Image, UnidentifiedImageError
except ModuleNotFoundError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass


OPENVERSE_IMAGES_URL = "https://api.openverse.org/v1/images/"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
GOOGLE_CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
BING_IMAGE_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"
BING_IMAGES_HTML_URL = "https://www.bing.com/images/search"
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/"
DUCKDUCKGO_IMAGES_URL = "https://duckduckgo.com/i.js"

DEFAULT_USER_AGENT = (
    "TrabalhoDOIVSuccessful3DPrintCollector/0.1 "
    "(academic computer vision dataset)"
)

SUCCESSFUL_PRINT_QUERIES = [
    "successful 3d print",
    "finished 3d print",
    "completed 3d print",
    "3d printed object",
    "3d printed model",
    "3d printed part",
    "3d print on build plate",
    "3d printer finished print",
    "3d printing object close up",
    "FDM 3d printed object",
    "PLA 3d printed object",
    "3DBenchy successful print",
    "impressao 3d bem sucedida",
    "objeto impresso em 3d",
    "peca impressa em 3d",
]

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@dataclass
class ImageCandidate:
    provider: str
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
        description="Baixa imagens aleatorias de impressoes 3D bem-sucedidas."
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Quantidade de imagens que devem ser baixadas.",
    )
    parser.add_argument(
        "--output",
        default="datasets/raw/successful_3d_prints",
        help="Pasta de saida para imagens e metadados.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=[
            "google",
            "bing",
            "duckduckgo",
            "openverse",
            "wikimedia",
        ],
        default=["duckduckgo", "bing", "openverse", "wikimedia"],
        help=(
            "Fontes usadas para busca. 'google' usa a API Google Custom Search; "
            "'bing' usa API se houver chave, senao busca HTML publica."
        ),
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Termos de busca opcionais. Se omitido, usa termos padrao.",
    )
    parser.add_argument("--results-per-query", type=int, default=30)
    parser.add_argument(
        "--max-random-page",
        type=int,
        default=8,
        help="Pagina/offset maximo usado para variar os resultados.",
    )
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--max-mb", type=float, default=8.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--google-api-key",
        default=os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        help="Chave da Google Custom Search API. Tambem aceita GOOGLE_CSE_API_KEY.",
    )
    parser.add_argument(
        "--google-cse-id",
        default=os.getenv("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CX"),
        help="ID do mecanismo Google Custom Search. Tambem aceita GOOGLE_CSE_ID.",
    )
    parser.add_argument(
        "--bing-api-key",
        default=os.getenv("BING_IMAGE_SEARCH_KEY") or os.getenv("BING_SEARCH_KEY"),
        help="Chave opcional da Bing Image Search API.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing = []
    if requests is None:
        missing.append("requests")
    if Image is None:
        missing.append("Pillow")

    if missing:
        raise SystemExit(
            "Dependencias ausentes: "
            + ", ".join(missing)
            + "\nInstale com: pip install -r requirements.txt"
        )


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", "", str(value))
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "image"


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Api-User-Agent": user_agent,
            "Accept": "application/json,image/*;q=0.9,*/*;q=0.8",
        }
    )
    return session


def search_openverse(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
) -> list[ImageCandidate]:
    page = random.randint(1, max(1, max_random_page))
    params = {"q": query, "page_size": min(limit, 50), "page": page}
    response = session.get(OPENVERSE_IMAGES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    candidates: list[ImageCandidate] = []
    for item in payload.get("results", []):
        image_url = item.get("url") or item.get("thumbnail")
        if not image_url:
            continue

        candidates.append(
            ImageCandidate(
                provider="openverse",
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

    random.shuffle(candidates)
    return candidates


def search_wikimedia(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
) -> list[ImageCandidate]:
    offset = random.randint(0, max(0, max_random_page - 1)) * min(limit, 50)
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "maxlag": 5,
    }
    response = session.get(WIKIMEDIA_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    candidates: list[ImageCandidate] = []
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

        candidates.append(
            ImageCandidate(
                provider="wikimedia",
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

    random.shuffle(candidates)
    return candidates


def search_google(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
    api_key: str,
    cse_id: str,
) -> list[ImageCandidate]:
    page = random.randint(0, max(0, max_random_page - 1))
    num = min(limit, 10)
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "searchType": "image",
        "num": num,
        "start": page * num + 1,
        "safe": "off",
    }
    response = session.get(GOOGLE_CUSTOM_SEARCH_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    candidates: list[ImageCandidate] = []
    for item in payload.get("items", []):
        image_url = item.get("link", "")
        if not image_url:
            continue

        image_info = item.get("image", {})
        candidates.append(
            ImageCandidate(
                provider="google",
                query=query,
                title=clean_text(item.get("title")),
                image_url=image_url,
                page_url=image_info.get("contextLink") or item.get("displayLink", ""),
                license_name="",
                license_url="",
                creator="",
                source=clean_text(item.get("displayLink") or "Google Custom Search"),
            )
        )

    random.shuffle(candidates)
    return candidates


def search_bing_api(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
    api_key: str,
) -> list[ImageCandidate]:
    count = min(limit, 50)
    offset = random.randint(0, max(0, max_random_page - 1)) * count
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": count,
        "offset": offset,
        "safeSearch": "Off",
        "imageType": "Photo",
    }
    response = session.get(
        BING_IMAGE_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    candidates: list[ImageCandidate] = []
    for item in payload.get("value", []):
        image_url = item.get("contentUrl", "")
        if not image_url:
            continue

        candidates.append(
            ImageCandidate(
                provider="bing_api",
                query=query,
                title=clean_text(item.get("name")),
                image_url=image_url,
                page_url=item.get("hostPageUrl") or "",
                license_name="",
                license_url="",
                creator="",
                source=clean_text(item.get("hostPageDisplayUrl") or "Bing Images"),
            )
        )

    random.shuffle(candidates)
    return candidates


def extract_bing_image_metadata(page_text: str) -> list[dict]:
    metadata_items: list[dict] = []
    for raw_metadata in re.findall(r'm="([^"]+)"', page_text):
        try:
            metadata = json.loads(html.unescape(raw_metadata))
        except json.JSONDecodeError:
            continue
        metadata_items.append(metadata)
    return metadata_items


def search_bing_html(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
) -> list[ImageCandidate]:
    offset = random.randint(0, max(0, max_random_page - 1)) * min(limit, 35) + 1
    params = {"q": query, "first": offset, "form": "HDRSC2"}
    response = session.get(BING_IMAGES_HTML_URL, params=params, timeout=timeout)
    response.raise_for_status()

    candidates: list[ImageCandidate] = []
    for metadata in extract_bing_image_metadata(response.text):
        image_url = metadata.get("murl", "")
        if not image_url:
            continue

        candidates.append(
            ImageCandidate(
                provider="bing_html",
                query=query,
                title=clean_text(metadata.get("t")),
                image_url=image_url,
                page_url=metadata.get("purl") or "",
                license_name="",
                license_url="",
                creator="",
                source=clean_text(metadata.get("purl") or "Bing Images"),
            )
        )

        if len(candidates) >= limit:
            break

    random.shuffle(candidates)
    return candidates


def extract_duckduckgo_token(page_text: str) -> str | None:
    patterns = [
        r"vqd='([^']+)'",
        r'vqd="([^"]+)"',
        r"vqd=([\d-]+)&",
        r"vqd=([^&]+)&",
    ]
    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            return html.unescape(match.group(1))
    return None


def search_duckduckgo(
    session: requests.Session,
    query: str,
    limit: int,
    max_random_page: int,
    timeout: float,
) -> list[ImageCandidate]:
    token_response = session.get(
        DUCKDUCKGO_SEARCH_URL,
        params={"q": query},
        timeout=timeout,
    )
    token_response.raise_for_status()
    token = extract_duckduckgo_token(token_response.text)
    if not token:
        print("  duckduckgo: token de busca nao encontrado")
        return []

    offset = random.randint(0, max(0, max_random_page - 1)) * min(limit, 100)
    params = {
        "l": "us-en",
        "o": "json",
        "q": query,
        "vqd": token,
        "f": ",,,",
        "p": "1",
        "s": offset,
    }
    response = session.get(DUCKDUCKGO_IMAGES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    candidates: list[ImageCandidate] = []
    for item in payload.get("results", []):
        image_url = item.get("image", "")
        if not image_url:
            continue

        candidates.append(
            ImageCandidate(
                provider="duckduckgo",
                query=query,
                title=clean_text(item.get("title")),
                image_url=image_url,
                page_url=item.get("url") or "",
                license_name="",
                license_url="",
                creator="",
                source=clean_text(item.get("source") or item.get("url")),
            )
        )

        if len(candidates) >= limit:
            break

    random.shuffle(candidates)
    return candidates


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
    candidate: ImageCandidate,
    output_dir: Path,
    existing_hashes: set[str],
    min_width: int,
    min_height: int,
    max_mb: float,
    timeout: float,
) -> DownloadRecord | None:
    response = session.get(candidate.image_url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        print(f"  ignorada: content-type nao suportado ({content_type})")
        return None

    content = response.content
    if len(content) > int(max_mb * 1024 * 1024):
        print(f"  ignorada: arquivo maior que {max_mb:.1f} MB")
        return None

    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 in existing_hashes:
        print("  ignorada: imagem duplicada")
        return None

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except UnidentifiedImageError:
        print("  ignorada: imagem invalida")
        return None

    width, height = image.size
    if width < min_width or height < min_height:
        print(f"  ignorada: resolucao baixa ({width}x{height})")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    extension = image_extension(image)
    file_name = f"successful_3d_print_{sha256[:12]}{extension}"
    file_path = output_dir / file_name

    if extension == ".jpg":
        image = image.convert("RGB")
        image.save(file_path, quality=95, optimize=True)
    else:
        image.save(file_path)

    existing_hashes.add(sha256)
    return DownloadRecord(
        file_path=str(file_path.as_posix()),
        sha256=sha256,
        width=width,
        height=height,
        provider=candidate.provider,
        query=candidate.query,
        title=candidate.title,
        image_url=candidate.image_url,
        page_url=candidate.page_url,
        license_name=candidate.license_name,
        license_url=candidate.license_url,
        creator=candidate.creator,
        source=candidate.source,
    )


def append_metadata(metadata_csv: Path, metadata_jsonl: Path, records: list[DownloadRecord]) -> None:
    if not records:
        return

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


def prepare_providers(
    providers: list[str],
    google_api_key: str | None,
    google_cse_id: str | None,
) -> list[str]:
    prepared = list(dict.fromkeys(providers))
    if "google" in prepared and (not google_api_key or not google_cse_id):
        print(
            "Google removido: informe --google-api-key e --google-cse-id "
            "ou configure GOOGLE_CSE_API_KEY e GOOGLE_CSE_ID."
        )
        prepared.remove("google")

    if not prepared:
        raise SystemExit("Nenhum provedor configurado para busca.")

    return prepared


def collect_candidates(
    session: requests.Session,
    providers: list[str],
    queries: list[str],
    results_per_query: int,
    max_random_page: int,
    timeout: float,
    google_api_key: str | None,
    google_cse_id: str | None,
    bing_api_key: str | None,
) -> list[ImageCandidate]:
    query = random.choice(queries)
    provider = random.choice(providers)
    print(f"[busca] {provider} | {query}")

    if provider == "google":
        return search_google(
            session=session,
            query=query,
            limit=results_per_query,
            max_random_page=max_random_page,
            timeout=timeout,
            api_key=google_api_key,
            cse_id=google_cse_id,
        )
    if provider == "bing":
        if bing_api_key:
            return search_bing_api(
                session=session,
                query=query,
                limit=results_per_query,
                max_random_page=max_random_page,
                timeout=timeout,
                api_key=bing_api_key,
            )
        return search_bing_html(
            session=session,
            query=query,
            limit=results_per_query,
            max_random_page=max_random_page,
            timeout=timeout,
        )
    if provider == "duckduckgo":
        return search_duckduckgo(
            session=session,
            query=query,
            limit=results_per_query,
            max_random_page=max_random_page,
            timeout=timeout,
        )
    if provider == "openverse":
        return search_openverse(
            session, query, results_per_query, max_random_page, timeout
        )
    if provider == "wikimedia":
        return search_wikimedia(
            session, query, results_per_query, max_random_page, timeout
        )
    return []


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    if args.count <= 0:
        raise SystemExit("--count deve ser maior que zero.")
    if args.seed is not None:
        random.seed(args.seed)

    output_dir = Path(args.output)
    metadata_csv = output_dir / "metadata.csv"
    metadata_jsonl = output_dir / "metadata.jsonl"
    queries = args.queries or SUCCESSFUL_PRINT_QUERIES

    session = make_session(args.user_agent)
    existing_urls, existing_hashes = read_existing_metadata(metadata_csv)
    seen_candidate_urls: set[str] = set()
    records: list[DownloadRecord] = []
    attempts = 0
    max_attempts = max(20, args.count * 12)
    providers = prepare_providers(
        args.providers,
        google_api_key=args.google_api_key,
        google_cse_id=args.google_cse_id,
    )

    print(f"Saida: {output_dir}")
    print(f"Quantidade desejada: {args.count}")
    print(f"Fontes: {', '.join(providers)}")
    if "bing" in providers and not args.bing_api_key:
        print("Bing sera usado pela busca HTML publica. Para API, informe --bing-api-key.")

    while len(records) < args.count and attempts < max_attempts:
        attempts += 1
        try:
            candidates = collect_candidates(
                session=session,
                providers=providers,
                queries=queries,
                results_per_query=args.results_per_query,
                max_random_page=args.max_random_page,
                timeout=args.timeout,
                google_api_key=args.google_api_key,
                google_cse_id=args.google_cse_id,
                bing_api_key=args.bing_api_key,
            )
        except requests.RequestException as error:
            print(f"  busca falhou: {error}")
            time.sleep(args.delay)
            continue

        for candidate in candidates:
            if len(records) >= args.count:
                break
            if candidate.image_url in existing_urls or candidate.image_url in seen_candidate_urls:
                continue
            seen_candidate_urls.add(candidate.image_url)

            if args.dry_run:
                print(f"  candidato: {candidate.title or candidate.image_url}")
                records.append(
                    DownloadRecord(
                        file_path="dry-run",
                        sha256="dry-run",
                        width=0,
                        height=0,
                        provider=candidate.provider,
                        query=candidate.query,
                        title=candidate.title,
                        image_url=candidate.image_url,
                        page_url=candidate.page_url,
                        license_name=candidate.license_name,
                        license_url=candidate.license_url,
                        creator=candidate.creator,
                        source=candidate.source,
                    )
                )
                continue

            try:
                record = download_image(
                    session=session,
                    candidate=candidate,
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

            existing_urls.add(candidate.image_url)
            records.append(record)
            print(f"  salva: {record.file_path} ({record.width}x{record.height})")
            time.sleep(args.delay)

    if not args.dry_run:
        append_metadata(metadata_csv, metadata_jsonl, records)

    print("\nColeta finalizada.")
    print(f"Imagens obtidas: {len(records)} de {args.count}")
    if len(records) < args.count:
        print("Aumente --max-random-page, --results-per-query ou relaxe os filtros.")
    if not args.dry_run:
        print(f"Metadados: {metadata_csv}")


if __name__ == "__main__":
    main()
