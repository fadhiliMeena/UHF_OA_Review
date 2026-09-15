#!/usr/bin/env python3
"""Download the live OpenAlex corpus and referenced-work metadata.

Uses the frozen OQL query in config/final_openalex_search.oql. Long OQL is sent
with POST to the API root and cursor pagination. API key is sent only in the
Authorization header and is never written to disk.
"""
from __future__ import annotations
import argparse, gzip, json, os, time
from pathlib import Path
import requests
from dotenv import load_dotenv

BASE = "https://api.openalex.org"


def request_json(session, method, url, *, headers, params=None, json_body=None, retries=8):
    for attempt in range(retries):
        try:
            r = session.request(method, url, headers=headers, params=params, json=json_body, timeout=90)
            if r.status_code == 200:
                return r.json(), r.headers
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(60, 2 ** attempt)
                print(f"OpenAlex returned {r.status_code}; retrying in {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenAlex error {r.status_code}: {r.text[:1000]}")
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(f"Network error: {exc}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("OpenAlex request failed after retries")


def download_oql(query: str, api_key: str, out: Path, manifest: Path, force=False):
    if out.exists() and manifest.exists() and not force:
        m = json.loads(manifest.read_text())
        if m.get("complete"):
            print(f"Using existing complete works download: {out}")
            return m
    out.parent.mkdir(parents=True, exist_ok=True)
    if force:
        out.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
    checkpoint = out.with_suffix(out.suffix + ".checkpoint.json")
    cursor = "*"; n = 0; pages = 0; expected = None; cost = 0.0
    mode = "wt"
    if out.exists() and checkpoint.exists() and not force:
        cp = json.loads(checkpoint.read_text())
        cursor = cp.get("next_cursor") or "*"; n = cp.get("records", 0); pages = cp.get("pages", 0)
        expected = cp.get("expected_count"); cost = cp.get("cost_usd", 0.0); mode = "at"
        print(f"Resuming at {n:,} works")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "HPP-UHC-financing-reproducibility/1.0"}
    with requests.Session() as s, gzip.open(out, mode, encoding="utf-8") as f:
        while cursor:
            body = {"oql": query, "per_page": 100, "cursor": cursor}
            data, _ = request_json(s, "POST", BASE + "/", headers=headers, json_body=body)
            results = data.get("results", [])
            meta = data.get("meta", {})
            if expected is None: expected = meta.get("count")
            cost += float(meta.get("cost_usd") or 0)
            for w in results:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
            n += len(results); pages += 1
            cursor = meta.get("next_cursor")
            checkpoint.write_text(json.dumps({"next_cursor": cursor, "records": n, "pages": pages, "expected_count": expected, "cost_usd": cost}, indent=2))
            if pages % 10 == 0 or not cursor:
                print(f"works: {n:,}/{expected or 0:,}  pages={pages}  API cost=${cost:.4f}")
            if not results:
                break
    m = {"complete": True, "records": n, "expected_count_at_download": expected, "pages": pages, "cost_usd": cost, "query_file": "config/final_openalex_search.oql"}
    manifest.write_text(json.dumps(m, indent=2))
    checkpoint.unlink(missing_ok=True)
    return m


def collect_reference_ids(works_gz: Path):
    ids = set()
    with gzip.open(works_gz, "rt", encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            for ref in w.get("referenced_works") or []:
                ids.add(ref.rsplit("/", 1)[-1])
    return sorted(ids)


def download_references(ids, api_key: str, out: Path, manifest: Path, force=False):
    if out.exists() and manifest.exists() and not force:
        m = json.loads(manifest.read_text())
        if m.get("complete"):
            print(f"Using existing complete reference download: {out}")
            return m
    if force:
        out.unlink(missing_ok=True); manifest.unlink(missing_ok=True)
    checkpoint = out.with_suffix(out.suffix + ".checkpoint.json")
    start = 0; n = 0; cost = 0.0; mode = "wt"
    if out.exists() and checkpoint.exists() and not force:
        cp = json.loads(checkpoint.read_text()); start = cp.get("next_index", 0); n = cp.get("records", 0); cost = cp.get("cost_usd", 0.0); mode = "at"
        print(f"Resuming referenced works at batch index {start:,}")
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "HPP-UHC-financing-reproducibility/1.0"}
    select = "id,title,publication_year,doi,authorships,locations,primary_location,cited_by_count"
    with requests.Session() as s, gzip.open(out, mode, encoding="utf-8") as f:
        for i in range(start, len(ids), 100):
            batch = "|".join(ids[i:i+100])
            params = {"filter": f"openalex:{batch}", "per_page": 100, "select": select}
            data, _ = request_json(s, "GET", BASE + "/works", headers=headers, params=params)
            meta = data.get("meta", {}); cost += float(meta.get("cost_usd") or 0)
            for w in data.get("results", []):
                f.write(json.dumps(w, ensure_ascii=False) + "\n"); n += 1
            checkpoint.write_text(json.dumps({"next_index": i+100, "records": n, "cost_usd": cost}, indent=2))
            if (i // 100 + 1) % 50 == 0:
                print(f"references: {min(i+100,len(ids)):,}/{len(ids):,} IDs queried; {n:,} metadata records")
    m = {"complete": True, "unique_reference_ids_queried": len(ids), "metadata_records_returned": n, "cost_usd": cost}
    manifest.write_text(json.dumps(m, indent=2)); checkpoint.unlink(missing_ok=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    repo = args.repo.resolve(); work = (args.workdir or repo / "work").resolve(); data = work / "data"; data.mkdir(parents=True, exist_ok=True)
    load_dotenv(repo / ".env")
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    if not key or key == "paste_your_key_here":
        raise SystemExit("OPENALEX_API_KEY is missing. Copy .env.example to .env and paste your OpenAlex key.")
    query = (repo / "config/final_openalex_search.oql").read_text(encoding="utf-8")
    works = data / "LLMIC_health_financing_OpenAlex_FULL_WORKS.jsonl.gz"
    wm = data / "works_download_manifest.json"
    download_oql(query, key, works, wm, force=args.force)
    ids = collect_reference_ids(works)
    (data / "referenced_work_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    refs = data / "LLMIC_health_financing_CITED_REFERENCES.jsonl.gz"
    rm = data / "references_download_manifest.json"
    download_references(ids, key, refs, rm, force=args.force)
    print("OpenAlex download complete.")

if __name__ == "__main__":
    main()
