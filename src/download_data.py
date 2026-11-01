"""Download the original CLINC150 files into the repository's data/ directory."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

SOURCE_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BASE_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{SOURCE_COMMIT}/data"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "data_full.json": "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa",
    "domains.json": "60a74358e52060e60b6128ae4efe679e9d6ba69c",
}


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, expected_sha in FILES.items():
        destination = DATA_DIR / filename
        if destination.exists() and git_blob_sha(destination.read_bytes()) == expected_sha:
            print(f"Already downloaded: {destination}")
            continue

        url = f"{BASE_URL}/{filename}"
        with urlopen(url, timeout=30) as response:
            content = response.read()

        if git_blob_sha(content) != expected_sha:
            raise ValueError(f"Checksum mismatch for {filename}")
        json.loads(content)
        destination.write_bytes(content)
        print(f"Downloaded: {destination}")


if __name__ == "__main__":
    main()
