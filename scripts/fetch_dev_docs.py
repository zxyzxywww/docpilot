"""DocPilot 文档抓取器:官方开发者文档 → HTML + manifest。

数据源(仅官方域名,内容权威可溯源):
    fastapi.tiangolo.com / docs.pydantic.dev / docs.sqlalchemy.org / docs.python.org

用法(需可访问外网的机器,如手机热点/家里网络):
    python scripts/fetch_dev_docs.py                     # 抓取清单全部页面
    python scripts/fetch_dev_docs.py --limit 10          # 只抓前 10 页(试跑)
    python scripts/fetch_dev_docs.py --force             # 忽略已下载,强制重抓

输出:
    data/raw/devdocs/<sha12>.html        按内容哈希命名,去重
    data/raw/devdocs/manifest.jsonl      一行一条:{url,title,domain,status,sha256,path,...}

特性:官方域名白名单、User-Agent、超时与指数退避重试、限流、
     content-type 校验、断点续传(已下载且 200 的跳过,除非 --force)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = PROJECT_ROOT / "sources" / "dev_docs.txt"
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "devdocs"

ALLOWED_DOMAINS = {
    "fastapi.tiangolo.com",
    "docs.pydantic.dev",
    "docs.sqlalchemy.org",
    "docs.python.org",
}
UA = "Mozilla/5.0 (compatible; DocPilot-fetch/1.0)"
TIMEOUT = httpx.Timeout(30.0, connect=15.0)
RATE_SECONDS = 0.3
MAX_RETRIES = 3


def extract_title(html: str, url: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:200]
    return url.rstrip("/").split("/")[-1]


def load_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def check_domain(url: str) -> str:
    domain = url.split("/")[2].lower()
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"非白名单域名: {domain}(仅允许 {sorted(ALLOWED_DOMAINS)})")
    return domain


def load_existing_manifest(out: Path) -> dict[str, dict]:
    """已成功下载的 URL → manifest 记录(断点续传)。"""
    index: dict[str, dict] = {}
    manifest_path = out / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == 200:
                    index[rec["url"]] = rec
    return index


def fetch_one(client: httpx.Client, url: str) -> tuple[str | None, str | None, int | None, int]:
    """下载一页。返回 (html, error, status, bytes)。"""
    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                if "text/html" not in ctype and "application/xhtml" not in ctype:
                    return None, f"非 HTML content-type: {ctype}", resp.status_code, len(resp.content)
                return resp.text, None, 200, len(resp.content)
            if resp.status_code in (403, 404, 451):
                return None, f"HTTP {resp.status_code}", resp.status_code, 0
            last_err = f"HTTP {resp.status_code}"
        except httpx.TimeoutException as exc:
            last_err = f"timeout: {type(exc).__name__}"
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}"
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 * (attempt + 1))
    return None, last_err, None, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", default=str(DEFAULT_LIST), help="URL 清单文件路径")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="仅抓前 N 页(试跑)")
    ap.add_argument("--force", action="store_true", help="忽略已下载强制重抓")
    ap.add_argument("--dry-run", action="store_true", help="只打印清单统计,不下载")
    args = ap.parse_args()

    urls = load_urls(Path(args.list))
    if args.limit:
        urls = urls[: args.limit]
    print(f"[dev_docs] 清单共 {len(urls)} 页")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    if args.dry_run:
        domains: dict[str, int] = {}
        for u in urls:
            d = u.split("/")[2]
            domains[d] = domains.get(d, 0) + 1
        for d, n in sorted(domains.items()):
            print(f"  {d}: {n}")
        return 0

    existing = {} if args.force else load_existing_manifest(out)
    print(f"[dev_docs] 已缓存 {len(existing)} 页,开始抓取…")

    failed: list[str] = []
    with httpx.Client(headers={"User-Agent": UA}, timeout=TIMEOUT, follow_redirects=True) as client:
        for i, url in enumerate(urls, 1):
            domain = check_domain(url)
            if url in existing and not args.force:
                continue
            html, error, status, size = fetch_one(client, url)
            rec = {
                "url": url,
                "domain": domain,
                "status": status,
                "title": "",
                "sha256": "",
                "path": "",
                "content_type": "",
                "size": size,
                "error": error,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if html is not None:
                raw_sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
                fname = f"{raw_sha[:12]}.html"
                (out / fname).write_text(html, encoding="utf-8")
                rec.update(
                    {
                        "status": 200,
                        "title": extract_title(html, url),
                        "sha256": raw_sha,
                        "path": fname,
                    }
                )
            else:
                failed.append(url)
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(urls)}] {status or 'ERR'} {domain} {rec.get('size', 0)}B  {url[:80]}")
            time.sleep(RATE_SECONDS)

    done = len(urls) - len(failed) + len([u for u in existing if u in urls])
    print(f"[dev_docs] 完成:成功(含缓存){done}/{len(urls)},失败 {len(failed)}")
    for u in failed:
        print(f"  FAIL {u}")
    return 1 if failed and not args.force and len(failed) == len(urls) else 0


if __name__ == "__main__":
    raise SystemExit(main())
