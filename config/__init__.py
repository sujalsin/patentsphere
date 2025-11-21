"""
Configuration management for PatentSphere.
Loads settings from config.yaml and environment variables.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
