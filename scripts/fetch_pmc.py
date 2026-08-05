"""PMC Open Access 文献下载与 manifest 生成(数据合规第一关)。

用法:
    python scripts/fetch_pmc.py --pmcids 13273211,13024797,13210980,13375924,12963003

仅下载明确允许复用的 PMC OA 文章的 JATS XML(合规红线:不碰版权不明 PDF,
不批量抓取普通 PMC 网页)。每篇生成确定性 document_id(sha256 前 12 位),
并写入 data/raw/manifest.jsonl(该文件是唯一入库的合规清单,原始 XML 不入 git)。

依赖:仅 Python 标准库(urllib / xml.etree / hashlib / json)。
兼容性:PMC XML 部分带 JATS 命名空间、部分不带,解析时自动探测。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"jats": "http://www.ncbi.nlm.nih.gov/JATS1"}
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}"


class FetchError(Exception):
    """下载或解析 PMC 文章失败。"""


def fetch_xml(pmc_id: str) -> bytes:
    """通过 E-utilities 下载 PMC 文章的 JATS XML。"""
    url = f"{EUTILS_BASE}/efetch.fcgi?db=pmc&id={pmc_id}&rettype=xml"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except Exception as exc:
        raise FetchError(f"PMC {pmc_id} 下载失败: {exc}") from exc


def _ns_for(root: ET.Element) -> dict[str, str]:
    """探测文档命名空间:根节点 tag 含 '{' 则使用 JATS 命名空间,否则空。"""
    return NS if "{" in root.tag else {}


def _find(root: ET.Element, path: str, ns: dict[str, str]) -> ET.Element | None:
    if not ns:
        path = path.replace("jats:", "")
    return root.find(path, ns)


def _findall(root: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    if not ns:
        path = path.replace("jats:", "")
    return list(root.findall(path, ns))


def _join_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def _text(root: ET.Element, path: str, ns: dict[str, str]) -> str:
    return _join_text(_find(root, path, ns))


def _authors(root: ET.Element, ns: dict[str, str]) -> list[str]:
    """提取作者列表(按出现顺序,过滤重复标记)。"""
    names: list[str] = []
    seen: set[str] = set()
    for contrib in _findall(root, ".//jats:contrib", ns):
        ctype = contrib.get("contrib-type")
        if ctype not in (None, "author", "aut"):
            continue
        name = _join_text(_find(contrib, "jats:name", ns)) or _join_text(
            _find(contrib, "jats:string-name", ns)
        )
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _license(root: ET.Element, ns: dict[str, str]) -> str:
    """提取 license 文本或链接。"""
    lic = _find(root, ".//jats:permissions/jats:license", ns)
    if lic is None:
        return ""
    text = _join_text(_find(lic, "jats:license-p", ns))
    href = lic.get("{http://www.w3.org/1999/xlink}href", "")
    return text or href


def _pub_date(root: ET.Element, ns: dict[str, str]) -> str:
    """优先取 date-type=pub 的日期,统一为 YYYY-MM-DD;否则取第一个有 year 的 pub-date。"""
    pubs = _findall(root, ".//jats:article-meta/jats:pub-date", ns)
    pubs.sort(key=lambda p: 0 if p.get("date-type") == "pub" else 1)
    for pub in pubs:
        year = _join_text(_find(pub, "jats:year", ns))
        if year:
            month = _join_text(_find(pub, "jats:month", ns)) or "01"
            day = _join_text(_find(pub, "jats:day", ns)) or "01"
            return f"{year}-{month}-{day}"
    for pub in pubs:
        text = _join_text(pub)
        if text:
            return text
    return ""


def _journal(root: ET.Element, ns: dict[str, str]) -> str:
    return _text(root, ".//jats:journal-title", ns)


def parse_metadata(xml_bytes: bytes, pmc_id: str) -> dict[str, str | list[str]]:
    """从 JATS XML 解析 manifest 所需元数据。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FetchError(f"PMC {pmc_id}: XML 解析失败: {exc}") from exc
    ns = _ns_for(root)
    article = _find(root, ".//jats:article", ns)
    if article is None:
        raise FetchError(f"PMC {pmc_id}: 未找到 article 节点")
    meta = _find(article, "jats:front/jats:article-meta", ns)
    if meta is None:
        meta = article
    return {
        "title": _text(meta, "title-group/article-title", ns),
        "authors": _authors(article, ns),
        "journal": _journal(article, ns),
        "doi": _text(meta, 'article-id[@pub-id-type="doi"]', ns),
        "license": _license(article, ns),
        "publication_date": _pub_date(article, ns),
        "document_type": article.get("article-type", "") or "research-article",
    }


def _load_existing(manifest_path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not manifest_path.exists():
        return existing
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            existing[rec["pmcid"]] = rec
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 PMC OA 文献并生成 manifest")
    parser.add_argument(
        "--pmcids", required=True, help="逗号分隔的 PMC ID 列表,如 13273211,13024797"
    )
    parser.add_argument("--out", default="data/raw", help="输出目录(默认 data/raw)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    existing = _load_existing(manifest_path)

    for raw_id in args.pmcids.split(","):
        pmc_id = raw_id.strip()
        if not pmc_id:
            continue
        if pmc_id in existing:
            print(f"跳过已存在: PMC{pmc_id}")
            continue
        xml_bytes = fetch_xml(pmc_id)
        meta = parse_metadata(xml_bytes, pmc_id)
        sha = hashlib.sha256(xml_bytes).hexdigest()
        doc_id = sha[:12]
        local_path = out / f"{doc_id}.xml"
        local_path.write_bytes(xml_bytes)
        rec: dict[str, object] = {
            "document_id": doc_id,
            "pmcid": pmc_id,
            "title": meta["title"],
            "authors": meta["authors"],
            "journal": meta["journal"],
            "doi": meta["doi"],
            "source_url": PMC_URL.format(pmcid=pmc_id),
            "license": meta["license"],
            "publication_date": meta["publication_date"],
            "document_type": meta["document_type"],
            "sha256": sha,
            "local_path": str(local_path),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        title = str(meta["title"])
        print(f"OK PMC{pmc_id}: {title[:55]} -> {doc_id}")
        time.sleep(0.4)  # NCBI 限流:不超过 3 请求/秒
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
