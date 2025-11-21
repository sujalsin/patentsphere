#!/usr/bin/env python3
"""
Utility script to print the active runtime profile and dataset settings.
Run this after activating your virtual environment to verify that the
application is operating in local_dev mode before touching any cloud resources.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.runtime_profile import get_runtime_profile  # type: ignore  # noqa


def main() -> None:
    profile = get_runtime_profile()
    print(f"Active profile      : {profile.profile}")
    print(f"Dataset path        : {profile.dataset_path()}")
    print(f"Adaptive Retrieval  : {profile.adaptive_retrieval_enabled()}")
    print(f"Critic Enabled      : {profile.critic_enabled()}")
    print("Cloud usage allowed :")
    print(f"  BigQuery exports  : {profile.allow_bigquery_exports()}")
    print(f"  VM start          : {profile.allow_vm_start()}")
    print(f"  Storage write     : {profile.allow_storage_write()}")


if __name__ == "__main__":
    main()

