"""Acquire only the predeclared training-month BTC pilot and source receipts."""

import json
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from spot_perp_basis.data.ingest import checksum_value
from spot_perp_basis.artifacts import ROOT, file_sha256, load_config, utc_now, write_json

MANIFEST = ROOT / "manifests/pilot_acquisition.json"


def fetch(url):
    request = Request(url, headers={"User-Agent": "SpotPerpBasisAcademicResearch/0.1"})
    with urlopen(request, timeout=25) as response:
        body = response.read()
        metadata = {"url": url, "final_url": response.url, "status": response.status,
                    "retrieved_at_utc": utc_now(), "http_headers": dict(response.headers.items())}
    return body, metadata


def save(body, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != body:
        raise ValueError(f"Refusing to replace an existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(body)
    temporary.replace(path)
    return {**metadata, "path": str(path.relative_to(ROOT)), "bytes": len(body),
            "sha256": file_sha256(path)}


def main():
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        restored = 0
        artifacts = [*existing["markets"].values(), *existing["documents"].values(),
                     *(market["checksum_receipt"] for market in existing["markets"].values())]
        for info in artifacts:
            path = ROOT / info["path"]
            if not path.exists():
                body, _ = fetch(info["url"])
                if sha256(body).hexdigest() != info["sha256"]:
                    raise ValueError(f"Remote artifact changed; refusing to alter frozen receipt: {info['url']}")
                save(body, path, {})
                restored += 1
            if file_sha256(path) != info["sha256"]:
                raise ValueError(f"Corrupt acquisition artifact: {info['path']}")
        for info in existing["markets"].values():
            expected = checksum_value((ROOT / info["checksum_path"]).read_bytes(), info["filename"])
            if expected != info["sha256"]:
                raise ValueError("Saved official checksum mismatch")
        print(f"Existing pilot acquisition verified; restored {restored} missing artifacts" if restored
              else "Existing pilot acquisition verified; no network requests made")
        return
    cfg = load_config("configs/pilot.yaml")
    contract = load_config()
    start = pd.Timestamp(cfg["month"] + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin()
    train_start, train_end = map(pd.Timestamp, contract["splits"]["train"])
    if not (train_start <= start < end <= train_end) or cfg["symbol"] != "BTCUSDT" or cfg["interval"] != "1m":
        raise ValueError("Pilot acquisition is restricted to a BTCUSDT 1m training month")
    folder = ROOT / "data/raw/binance/pilot" / cfg["month"]
    inventory = json.loads((ROOT / "manifests/reused_spot_archives.json").read_text())
    markets = {}
    for market in ("spot", "usdm_perpetual"):
        filename = f"{cfg['symbol']}-1m-{cfg['month']}.zip"
        base = contract["markets"][market]["archive_base_url"]
        url = f"{base}/{cfg['symbol']}/1m/{filename}"
        check_body, check_meta = fetch(url + ".CHECKSUM")
        expected = checksum_value(check_body, filename)
        candidates = [row for row in inventory["files"] if market == "spot" and
                      row["symbol"] == cfg["symbol"] and row["month"] == cfg["month"] and row["sha256"] == expected]
        destination = ROOT / "data/raw/binance" / market / cfg["symbol"] / cfg["month"] / expected / filename
        if candidates and (ROOT / candidates[0]["path"]).exists():
            row = candidates[0]
            destination = ROOT / row["path"]
            if file_sha256(destination) != expected:
                raise ValueError("Corrupt reusable spot archive")
            metadata = {"url": url, "path": row["path"], "sha256": expected,
                        "bytes": row["bytes"], "acquisition": "reused_verified_spot_copy",
                        "original_downloaded_at_utc": row["original_downloaded_at_utc"]}
        else:
            body, meta = fetch(url)
            if sha256(body).hexdigest() != expected:
                raise ValueError(f"Official archive checksum mismatch: {market}")
            metadata = save(body, destination, meta)
            metadata["acquisition"] = "downloaded"
        check = save(check_body, folder / f"{market}.CHECKSUM", check_meta)
        metadata.update(market=market, symbol=cfg["symbol"], month=cfg["month"], interval="1m",
                        filename=filename, checksum_path=check["path"], checksum_receipt=check)
        markets[market] = metadata
        print(f"Verified {market}: {metadata['bytes']:,} bytes, SHA256 {expected}", flush=True)
    documents = {}
    for name, url in {
        "public_schema.md": "https://raw.githubusercontent.com/binance/binance-public-data/master/README.md",
        "spot_api_schema.md": "https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md",
    }.items():
        body, meta = fetch(url)
        documents[name] = save(body, folder / name, meta)
    # Exchange metadata establishes current product identity, not historical trading availability.
    identity = {}
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        body, meta = fetch(url)
        symbols = json.loads(body)["symbols"]
        selected = next(row for row in symbols if row["symbol"] == cfg["symbol"])
        if selected["contractType"] != "PERPETUAL" or selected["quoteAsset"] != "USDT":
            raise ValueError("Unexpected perpetual contract identity")
        documents["exchange_info.json"] = save(body, folder / "exchange_info.json", meta)
        identity = {key: selected[key] for key in ("symbol", "contractType", "baseAsset", "quoteAsset", "marginAsset")}
    except (HTTPError, URLError) as error:
        identity = {"url": url, "access_error": str(error),
                    "limitation": "Product identified by official USD-M archive path and undated BTCUSDT symbol; current API unavailable"}
    write_json(MANIFEST, {"created_at_utc": utc_now(), "config_sha256": file_sha256(ROOT / "configs/pilot.yaml"),
                          "markets": markets, "documents": documents, "current_product_identity": identity})


if __name__ == "__main__":
    main()
