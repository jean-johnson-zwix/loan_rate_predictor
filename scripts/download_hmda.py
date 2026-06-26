"""Download AZ HMDA Modified LAR CSVs (2021-2024) from the HMDA Data Browser API."""

import urllib.request
from pathlib import Path

BASE_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
YEARS = [2021, 2022, 2023, 2024]
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def download(year: int) -> None:
    url = f"{BASE_URL}?states=AZ&years={year}&actions_taken=1"
    dest = OUT_DIR / f"{year}.csv"
    if dest.exists():
        print(f"{year}: already exists, skipping")
        return
    print(f"{year}: downloading → {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):  # 1 MB chunks
            f.write(chunk)
    print(f"{year}: done ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        download(year)
