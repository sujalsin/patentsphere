"""
Runtime profile helpers for PatentSphere.

This module wires the YAML config + environment variables into a small
runtime policy object that other parts of the codebase (e.g. FastAPI startup,
data pipeline scripts) can use to *safely* decide when cloud resources
are allowed to be touched.

Key ideas:
- Default profile is `local_dev` → NO GCP usage.
- GCP usage requires BOTH:
  - A cloud-oriented profile (e.g. `gcp_free_tier`), AND
  - An explicit environment flag (e.g. GCP_ALLOW_VM_START=true).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from .settings import get_settings, load_yaml_config
from pathlib import Path


class RuntimeProfile:
    """
    High-level view over config + env for cost-safe decisions.
    """

    def __init__(self, settings, raw_config: Dict[str, Any]) -> None:
        self.settings = settings
        self.raw = raw_config

        self._app = self.raw.get("app", {}) or {}
        self._gcp = self.raw.get("gcp", {}) or {}
        self._gcp_usage = self._gcp.get("usage_policy", {}) or {}
        self._resource_policy = self.raw.get("resource_policy", {}) or {}
        self._profiles = self._resource_policy.get("profiles", {}) or {}

    # ------------------------------------------------------------------
    # Basic profile helpers
    # ------------------------------------------------------------------
    @property
    def profile(self) -> str:
        """Return the active high-level profile (e.g. local_dev)."""
        return (
            getattr(self.settings, "app_profile", None)
            or self._app.get("profile")
            or "local_dev"
        )

    def is_local(self) -> bool:
        return self.profile == "local_dev"

    def is_gcp_free_tier(self) -> bool:
        return self.profile == "gcp_free_tier"

    def is_gcp_full(self) -> bool:
        return self.profile == "gcp_full"

    # ------------------------------------------------------------------
    # Resource policy helpers
    # ------------------------------------------------------------------
    def profile_policy(self) -> Dict[str, Any]:
        """Return the resource-policy block for the active profile."""
        return self._profiles.get(self.profile, {}) or {}

    def max_parallel_agents(self) -> int:
        return int(self.profile_policy().get("max_parallel_agents", 2))

    def adaptive_retrieval_enabled(self) -> bool:
        return bool(self.profile_policy().get("enable_adaptive_retrieval", False))

    def critic_enabled(self) -> bool:
        return bool(self.profile_policy().get("enable_critic", False))

    def dataset_path(self) -> str:
        """
        Resolve the dataset path for this profile.

        - For local_dev → use data.subsets.local_subset_path
        - For GCP profiles → use data.subsets.full_corpus_path
        """
        data_cfg = self.raw.get("data", {}) or {}
        subsets = data_cfg.get("subsets", {}) or {}

        if self.is_local():
            return subsets.get("local_subset_path", "data/patents_50k.jsonl")
        # For any GCP profile, prefer full corpus path
        return subsets.get("full_corpus_path", "gs://patentsphere-data/patents_1m.jsonl")

    # ------------------------------------------------------------------
    # GCP safety gates
    # ------------------------------------------------------------------
    def _env_flag_true(self, var_name: str) -> bool:
        val = os.getenv(var_name, "").strip().lower()
        return val in {"1", "true", "yes", "on"}

    def _usage_flag_true(self, key: str) -> bool:
        return bool(self._gcp_usage.get(key, False))

    def allow_vm_start(self) -> bool:
        """
        Is it permissible to start a GCP VM?

        Requires:
        - usage_policy.allow_vm_start == true
        - GCP_ALLOW_VM_START env flag set true
        """
        return self._usage_flag_true("allow_vm_start") and self._env_flag_true(
            "GCP_ALLOW_VM_START"
        )

    def allow_bigquery_exports(self) -> bool:
        """
        Is it permissible to run non-dry-run BigQuery jobs / exports?

        Requires:
        - usage_policy.allow_bigquery_exports == true
        - GCP_ALLOW_BIGQUERY_EXPORTS env flag set true
        """
        return self._usage_flag_true(
            "allow_bigquery_exports"
        ) and self._env_flag_true("GCP_ALLOW_BIGQUERY_EXPORTS")

    def allow_storage_write(self) -> bool:
        """
        Is it permissible to write objects to GCS?
        """
        return self._usage_flag_true("allow_storage_write") and self._env_flag_true(
            "GCP_ALLOW_STORAGE_WRITE"
        )

    def require_manual_cloud_enable(self) -> bool:
        return bool(self._resource_policy.get("require_manual_cloud_enable", True))

    def fail_if_gcp_usage_without_flag(self) -> bool:
        return bool(self._resource_policy.get("fail_if_gcp_usage_without_flag", True))

    # ------------------------------------------------------------------
    # Guard helpers to be used by API startup / scripts
    # ------------------------------------------------------------------
    def ensure_gcp_vm_allowed(self) -> None:
        if not self.allow_vm_start() and self.fail_if_gcp_usage_without_flag():
            raise RuntimeError(
                "GCP VM usage blocked by config.\n"
                "To enable, you must:\n"
                "  1) Set app.profile to 'gcp_free_tier' or 'gcp_full' in config.yaml, AND\n"
                "  2) Export GCP_ALLOW_VM_START=true in your environment.\n"
                "This prevents accidental credit burn from local runs."
            )

    def ensure_bigquery_export_allowed(self) -> None:
        if not self.allow_bigquery_exports() and self.fail_if_gcp_usage_without_flag():
            raise RuntimeError(
                "BigQuery export / non-dry-run jobs are disabled by config.\n"
                "To enable, you must:\n"
                "  1) Set app.profile to 'gcp_free_tier' or 'gcp_full', AND\n"
                "  2) Export GCP_ALLOW_BIGQUERY_EXPORTS=true.\n"
                "Dry-run queries are still allowed and recommended for cost checks."
            )

    def ensure_storage_write_allowed(self) -> None:
        if not self.allow_storage_write() and self.fail_if_gcp_usage_without_flag():
            raise RuntimeError(
                "Writing to GCS is disabled by config.\n"
                "To enable, you must:\n"
                "  1) Set app.profile to 'gcp_free_tier' or 'gcp_full', AND\n"
                "  2) Export GCP_ALLOW_STORAGE_WRITE=true.\n"
                "Keep this off unless you explicitly need cloud persistence."
            )


@lru_cache()
def get_runtime_profile() -> RuntimeProfile:
    """
    Load Settings + raw YAML and return a cached RuntimeProfile.

    This is the main entrypoint that other modules should import:

    from config.runtime_profile import get_runtime_profile
    rp = get_runtime_profile()
    if rp.is_local():
        ...
    """
    settings = get_settings()
    config_path = Path(__file__).parent / "config.yaml"
    raw_config = load_yaml_config(config_path)
    return RuntimeProfile(settings=settings, raw_config=raw_config)


__all__ = ["RuntimeProfile", "get_runtime_profile"]




