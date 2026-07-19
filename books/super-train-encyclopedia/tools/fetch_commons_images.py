#!/usr/bin/env python3
"""Fetch freely licensed Wikimedia Commons images for the train encyclopedia.

Targets live in ``image_targets.json`` beside this script.  For each target we
first ask English Wikipedia for the article's lead image, then fall back to a
Commons file search.  Only Creative Commons, CC0, or public-domain files are
accepted.  Images are resized without cropping and converted to WebP; complete
provenance is written to ``image_metadata.json`` and ``IMAGE_CREDITS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


BOOK_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
TARGETS_PATH = TOOLS_DIR / "image_targets.json"
OUTPUT_DIR = BOOK_DIR / "images" / "trains"
METADATA_PATH = BOOK_DIR / "image_metadata.json"
CREDITS_PATH = BOOK_DIR / "IMAGE_CREDITS.md"

USER_AGENT = (
    "StochasticParrotTrainEncyclopedia/1.0 "
    "(educational image attribution; local build script)"
)
ENWIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
LAST_REQUEST_AT = 0.0
REQUEST_INTERVAL = 0.35


def open_with_backoff(request: urllib.request.Request, timeout: int) -> Any:
    global LAST_REQUEST_AT
    for attempt in range(7):
        delay = REQUEST_INTERVAL - (time.monotonic() - LAST_REQUEST_AT)
        if delay > 0:
            time.sleep(delay)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            LAST_REQUEST_AT = time.monotonic()
            return response
        except urllib.error.HTTPError as exc:
            LAST_REQUEST_AT = time.monotonic()
            if exc.code != 429 or attempt == 6:
                raise
            retry_header = exc.headers.get("Retry-After", "")
            try:
                retry_after = float(retry_header)
            except ValueError:
                retry_after = 0.0
            wait = min(45.0, max(retry_after, 2.0 * (attempt + 1)))
            print(f"  rate limited; retry in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def api_get(endpoint: str, **params: Any) -> dict[str, Any]:
    params.update({"action": "query", "format": "json", "formatversion": 2})
    url = endpoint + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_with_backoff(request, timeout=40) as response:
        return json.load(response)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_with_backoff(request, timeout=60) as response:
        return response.read()


def plain_text(value: str | None) -> str:
    if not value:
        return "未注明"
    value = re.sub(r"<br\s*/?>", "; ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    repeated = re.fullmatch(r"(.{4,}?)\1", value)
    if repeated:
        value = repeated.group(1)
    return value or "未注明"


def meta_value(metadata: dict[str, Any], key: str, default: str = "") -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        return str(value.get("value", default))
    return default


def allowed_license(metadata: dict[str, Any]) -> bool:
    short = meta_value(metadata, "LicenseShortName").casefold()
    normalized = re.sub(r"[_-]+", " ", short)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = set(normalized.split())
    if {"nc", "nd"} & tokens or "noncommercial" in normalized or "no derivatives" in normalized:
        return False
    if normalized in {"public domain", "pdm"}:
        return True
    if re.fullmatch(r"cc0(?: [0-9.]+)?", normalized):
        return True
    return bool(
        re.fullmatch(
            r"cc by(?: sa)?(?: [0-9.]+)?(?: [a-z]{2}| international)?",
            normalized,
        )
    )


def commons_info(file_title: str, width: int) -> dict[str, Any] | None:
    if not file_title.lower().startswith("file:"):
        file_title = "File:" + file_title
    data = api_get(
        COMMONS_API,
        prop="imageinfo",
        iiprop="url|mime|size|extmetadata",
        iiurlwidth=width,
        titles=file_title,
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    infos = pages[0].get("imageinfo", [])
    if not infos:
        return None
    info = infos[0]
    if not str(info.get("mime", "")).startswith("image/"):
        return None
    metadata = info.get("extmetadata", {})
    if not allowed_license(metadata):
        return None
    return {
        "file_title": pages[0]["title"],
        "download_url": info.get("thumburl") or info.get("url"),
        "original_url": info.get("url"),
        "description_url": info.get("descriptionurl"),
        "mime": info.get("mime"),
        "original_width": info.get("width"),
        "original_height": info.get("height"),
        "metadata": metadata,
    }


def chunked(values: list[str], size: int = 40) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def title_key(value: str) -> str:
    return value.replace("_", " ").strip().casefold()


def wikipedia_lead_files(articles: list[str]) -> dict[str, str]:
    """Resolve article lead-image names in a few batched API requests."""
    output: dict[str, str] = {}
    for batch in chunked(list(dict.fromkeys(articles))):
        data = api_get(
            ENWIKI_API,
            prop="pageimages",
            piprop="name",
            redirects=1,
            titles="|".join(batch),
        )
        query = data.get("query", {})
        aliases: dict[str, str] = {}
        for item in query.get("normalized", []):
            aliases[title_key(item["from"])] = item["to"]
        for item in query.get("redirects", []):
            aliases[title_key(item["from"])] = item["to"]
        pages = {
            title_key(page.get("title", "")): page
            for page in query.get("pages", [])
            if not page.get("missing")
        }
        for article in batch:
            resolved = article
            for _ in range(4):
                next_title = aliases.get(title_key(resolved))
                if not next_title:
                    break
                resolved = next_title
            page = pages.get(title_key(resolved)) or pages.get(title_key(article))
            if page and page.get("pageimage"):
                output[article] = page["pageimage"]
    return output


def commons_infos(file_titles: list[str], width: int) -> dict[str, dict[str, Any]]:
    """Resolve Commons metadata for many exact file titles in batches."""
    output: dict[str, dict[str, Any]] = {}
    unique = list(dict.fromkeys(file_titles))
    for batch in chunked(unique):
        titles = [title if title.lower().startswith("file:") else "File:" + title for title in batch]
        data = api_get(
            COMMONS_API,
            prop="imageinfo",
            iiprop="url|mime|size|extmetadata",
            iiurlwidth=width,
            titles="|".join(titles),
        )
        for page in data.get("query", {}).get("pages", []):
            infos = page.get("imageinfo", [])
            if page.get("missing") or not infos:
                continue
            info = infos[0]
            if not str(info.get("mime", "")).startswith("image/"):
                continue
            metadata = info.get("extmetadata", {})
            if not allowed_license(metadata):
                continue
            record = {
                "file_title": page["title"],
                "download_url": info.get("thumburl") or info.get("url"),
                "original_url": info.get("url"),
                "description_url": info.get("descriptionurl"),
                "mime": info.get("mime"),
                "original_width": info.get("width"),
                "original_height": info.get("height"),
                "metadata": metadata,
            }
            output[title_key(page["title"].removeprefix("File:"))] = record
    return output


def resolve_targets(targets: list[dict[str, Any]], width: int) -> dict[str, dict[str, Any]]:
    articles = [
        target["article"]
        for target in targets
        if target.get("article") and not target.get("prefer_search") and not target.get("commons_file")
    ]
    leads = wikipedia_lead_files(articles)
    candidates: dict[str, str] = {}
    for target in targets:
        file_title = target.get("commons_file") or leads.get(target.get("article", ""))
        if file_title:
            candidates[target["id"]] = file_title
    info_by_file = commons_infos(list(candidates.values()), width)

    resolved: dict[str, dict[str, Any]] = {}
    for target in targets:
        candidate = candidates.get(target["id"])
        info = info_by_file.get(title_key(candidate)) if candidate else None
        if info:
            info = dict(info)
            info["selection"] = (
                "exact Commons file" if target.get("commons_file") else f"lead image: {target.get('article')}"
            )
            resolved[target["id"]] = info
            continue
        search_terms = [target.get("search"), target.get("article"), target["subject"]]
        for term in search_terms:
            if not term:
                continue
            info = commons_search(str(term), width)
            if info:
                info["selection"] = f"Commons search: {term}"
                resolved[target["id"]] = info
                break
    return resolved


def wikipedia_lead_file(article: str) -> str | None:
    data = api_get(
        ENWIKI_API,
        prop="pageimages",
        piprop="name",
        redirects=1,
        titles=article,
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    return pages[0].get("pageimage")


def commons_search(query: str, width: int) -> dict[str, Any] | None:
    data = api_get(
        COMMONS_API,
        generator="search",
        gsrsearch=query,
        gsrnamespace=6,
        gsrlimit=12,
        prop="imageinfo",
        iiprop="url|mime|size|extmetadata",
        iiurlwidth=width,
    )
    for page in data.get("query", {}).get("pages", []):
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        info = infos[0]
        if not str(info.get("mime", "")).startswith("image/"):
            continue
        metadata = info.get("extmetadata", {})
        if not allowed_license(metadata):
            continue
        title_lower = page.get("title", "").lower()
        if any(token in title_lower for token in ("logo", "map", "diagram")):
            continue
        return {
            "file_title": page["title"],
            "download_url": info.get("thumburl") or info.get("url"),
            "original_url": info.get("url"),
            "description_url": info.get("descriptionurl"),
            "mime": info.get("mime"),
            "original_width": info.get("width"),
            "original_height": info.get("height"),
            "metadata": metadata,
        }
    return None


def choose_image(target: dict[str, Any], width: int) -> dict[str, Any] | None:
    exact_file = target.get("commons_file")
    if exact_file:
        info = commons_info(exact_file, width)
        if info:
            info["selection"] = "exact Commons file"
            return info

    if not target.get("prefer_search") and target.get("article"):
        lead = wikipedia_lead_file(target["article"])
        if lead:
            info = commons_info(lead, width)
            if info:
                info["selection"] = f"lead image: {target['article']}"
                return info

    search_terms = [target.get("search"), target.get("article"), target["subject"]]
    for term in search_terms:
        if not term:
            continue
        info = commons_search(str(term), width)
        if info:
            info["selection"] = f"Commons search: {term}"
            return info
    return None


def save_webp(data: bytes, output: Path, max_width: int) -> tuple[int, int, str]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, "WEBP", quality=84, method=6)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return image.width, image.height, digest


def md_safe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def write_credits(records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# 图片来源与许可",
        "",
        "本书正文、`images/cover-super-train-encyclopedia.webp` 封面插画与原创 SVG 原理图按仓库的 CC0 许可发布。",
        "`images/trains/` 中的车型图片**不适用仓库的 CC0**；每张图片保留",
        "下表所列的原作者信息及许可或公有领域状态。需要署名的图片须按其许可署名。",
        "图片只做了等比例缩小与 WebP 格式转换，",
        "没有裁切；这项技术处理记为“已调整尺寸与格式”。",
        "",
        "重新使用图片前，请打开原文件页核对最新许可说明。Wikimedia Commons",
        "也提醒使用者自行确认文件页中的作者、许可与其他可能适用的权利。",
        "",
        "## 车型图片署名表",
        "",
        "| 图号 | 主题 | 作者/来源 | 许可/版权状态 | Commons 原文件页 | 本地文件 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in sorted(records, key=lambda item: item["id"]):
        anchor = f'<a id="img-{record["id"]}"></a>'
        source = f'[{md_safe(record["file_title"])}]({record["description_url"]})'
        creator = md_safe(record.get("attribution") or record["artist"])
        license_name = md_safe(record["license"])
        if record.get("license_url"):
            license_name = f'[{license_name}]({record["license_url"]})'
        lines.append(
            "| "
            + " | ".join(
                [
                    anchor + record["id"],
                    md_safe(record["subject"]),
                    creator,
                    license_name,
                    source,
                    f'`images/trains/{record["id"]}.webp`',
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 构建说明",
            "",
            "- 图片元数据快照见 [`image_metadata.json`](image_metadata.json)。",
            "- 获取脚本见 [`tools/fetch_commons_images.py`](tools/fetch_commons_images.py)。",
            "- 如需重新运行获取脚本，使用 Python 3，并先安装 [`tools/requirements.txt`](tools/requirements.txt) 中的 Pillow。",
            "- Wikimedia Commons 的站外复用说明：",
            "  [Commons:Reusing content outside Wikimedia](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/en)。",
        ]
    )
    if failures:
        lines.extend(["", "## 尚未取得图片的条目", ""])
        for failure in failures:
            lines.append(f'- `{failure["id"]}`：{failure["subject"]}')
    CREDITS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist(records: list[dict[str, Any]], previous: dict[str, dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    current_ids = {item["id"] for item in records}
    snapshot = records + [item for key, item in previous.items() if key not in current_ids]
    snapshot.sort(key=lambda item: item["id"])
    METADATA_PATH.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_credits(snapshot, failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--pause", type=float, default=0.08)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="process only the next N targets that lack saved metadata",
    )
    args = parser.parse_args()

    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    selected = set(args.only)
    if selected:
        targets = [target for target in targets if target["id"] in selected]

    previous: dict[str, dict[str, Any]] = {}
    if METADATA_PATH.exists():
        for item in json.loads(METADATA_PATH.read_text(encoding="utf-8")):
            previous[item["id"]] = item

    if args.batch_size and not args.refresh:
        targets = [
            target
            for target in targets
            if target["id"] not in previous
            or not (OUTPUT_DIR / f'{target["id"]}.webp').exists()
        ][: args.batch_size]

    print(f"resolve metadata for {len(targets)} targets")
    resolved = resolve_targets(targets, args.width)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        output = OUTPUT_DIR / f'{target["id"]}.webp'
        if output.exists() and target["id"] in previous and not args.refresh:
            records.append({**previous[target["id"]], **target})
            print(f"[{index}/{len(targets)}] keep {target['id']}")
            continue

        print(f"[{index}/{len(targets)}] fetch {target['id']}: {target['subject']}")
        try:
            info = resolved.get(target["id"])
            if not info or not info.get("download_url"):
                failures.append(target)
                print("  no suitable freely licensed image", file=sys.stderr)
                continue
            raw = download(info["download_url"])
            width, height, digest = save_webp(raw, output, args.width)
            metadata = info["metadata"]
            license_url = meta_value(metadata, "LicenseUrl")
            artist = plain_text(meta_value(metadata, "Artist"))
            credit = plain_text(meta_value(metadata, "Credit"))
            attribution = plain_text(meta_value(metadata, "Attribution"))
            if attribution == "未注明":
                attribution = artist
            if "German Federal Archive" in credit and "German Federal Archive" not in attribution:
                archive_match = re.search(r"Bundesarchiv Bild ([^,]+)", info["file_title"])
                archive_id = f", Bild {archive_match.group(1)}" if archive_match else ""
                attribution = f"{artist}; German Federal Archive (Bundesarchiv){archive_id}"
            record = {
                **target,
                "file_title": info["file_title"],
                "description_url": info["description_url"],
                "original_url": info["original_url"],
                "download_url": info["download_url"],
                "selection": info["selection"],
                "artist": artist,
                "credit": credit,
                "attribution": attribution,
                "license": plain_text(meta_value(metadata, "LicenseShortName")),
                "license_url": license_url,
                "date_time_original": plain_text(meta_value(metadata, "DateTimeOriginal")),
                "local_width": width,
                "local_height": height,
                "sha256": digest,
                "technical_changes": "resized proportionally and converted to WebP; not cropped",
            }
            records.append(record)
            persist(records, previous, failures)
            time.sleep(args.pause)
        except Exception as exc:  # keep the batch useful when one remote file fails
            failures.append(target)
            print(f"  ERROR: {exc}", file=sys.stderr)

    records.sort(key=lambda item: item["id"])
    persist(records, previous, failures)
    print(f"saved {len(records)} images; {len(failures)} unresolved")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
