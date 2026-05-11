"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage YAML configuration files."""

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return config.get('models', {}).get(model_name, {})

    @staticmethod
    def get_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get global configuration settings."""
        return config.get('global', {})
