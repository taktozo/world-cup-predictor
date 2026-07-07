"""Download the World Cup / international match dataset from Kaggle.

Requires a Kaggle API token saved at ~/.kaggle/access_token (or the
KAGGLE_API_TOKEN environment variable set). See kaggle.com/settings > API.
"""

import subprocess
from pathlib import Path

DATASETS = [
    "martj42/international-football-results-from-1872-to-2017",
    "saifalnimri/international-football-elo-ratings",
]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for dataset in DATASETS:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
