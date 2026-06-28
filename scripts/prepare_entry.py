"""SageMaker Processing entrypoint for the prepare step."""
import os
import sys

_SM_CODE = "/opt/ml/processing/input/source"
if os.path.isdir(_SM_CODE) and _SM_CODE not in sys.path:
    sys.path.insert(0, _SM_CODE)

import argparse
from pathlib import Path

from loan_rate_predictor.training.prepare import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-year", type=int, required=True)
    args = parser.parse_args()

    input_dir = Path("/opt/ml/processing/input/data")
    output_dir = Path("/opt/ml/processing/output")

    stats = prepare(input_dir, output_dir, args.data_year)
    print(f"Prepare complete: {stats}")


if __name__ == "__main__":
    main()
