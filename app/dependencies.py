from functools import lru_cache

from fastapi import Depends

from config.runtime_profile import get_runtime_profile, RuntimeProfile
from config.settings import get_settings, Settings


@lru_cache()
def get_app_settings() -> Settings:
    return get_settings()


@lru_cache()
def get_profile() -> RuntimeProfile:
    return get_runtime_profile()


def require_local_profile(profile: RuntimeProfile = Depends(get_profile)) -> RuntimeProfile:
    if not profile.is_local():
        raise RuntimeError("API is limited to local_dev profile until cloud guardrails are configured.")
    return profile


