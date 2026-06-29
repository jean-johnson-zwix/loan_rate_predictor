"""Update infra/terraform.auto.tfvars with the current champion's ARN and trained_on year.

Run this after a pipeline promotion to keep the pricing Lambda's MODEL_VINTAGE and the
endpoint's model_package_arn in sync. Promotion isn't done until this script has run
and `make tf-apply` has deployed.

Usage:
    python scripts/deploy_champion.py [--dry-run]
"""
import argparse
import os
import re
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loan_rate_predictor import config
from loan_rate_predictor.registry import resolve_champion

TFVARS = os.path.join(os.path.dirname(__file__), "..", "infra", "terraform.auto.tfvars")


def _read_tfvars(path: str) -> str:
    with open(path) as f:
        return f.read()


def _set_var(content: str, key: str, value: str) -> str:
    """Update or append a string variable in tfvars content."""
    pattern = rf'^{re.escape(key)}\s*=\s*"[^"]*"'
    replacement = f'{key}="{value}"'
    updated, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if n == 0:
        updated = content.rstrip("\n") + f'\n{replacement}\n'
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    session = boto3.Session(profile_name=config.AWS_PROFILE)
    sm = session.client("sagemaker", region_name=config.AWS_REGION)

    result = resolve_champion(sm)
    if result is None:
        print("No approved champion found in registry.")
        sys.exit(1)

    champion_arn, _ = result
    desc = sm.describe_model_package(ModelPackageName=champion_arn)
    metadata = desc.get("CustomerMetadataProperties", {})
    trained_on = metadata.get("trained_on")
    if not trained_on:
        print(f"Champion {champion_arn} has no 'trained_on' metadata tag. Tag it manually or re-register.")
        sys.exit(1)

    content = _read_tfvars(TFVARS)
    updated = _set_var(content, "model_package_arn", champion_arn)
    updated = _set_var(updated, "trained_on", trained_on)

    print(f"Champion ARN : {champion_arn}")
    print(f"trained_on   : {trained_on}")

    if args.dry_run:
        print("\n--- terraform.auto.tfvars (preview) ---")
        print(updated)
        return

    with open(TFVARS, "w") as f:
        f.write(updated)

    print(f"\nUpdated {TFVARS}")
    print("Run `make tf-apply` to deploy the new champion.")


if __name__ == "__main__":
    main()
