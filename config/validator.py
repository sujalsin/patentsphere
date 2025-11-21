"""
Configuration Validation Module

Validates config.yaml and .env files to ensure:
1. All required settings are present
2. GCP safeguards are properly configured
3. Local development is properly set up
4. Cost controls are in place
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from dotenv import load_dotenv


class ConfigValidator:
    """Validates PatentSphere configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.config: Dict = {}
        
    def load_config(self) -> bool:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Config file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False
    
    def check_env_file(self) -> bool:
        """Check if .env file exists and is properly configured"""
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if not env_file.exists():
            self.warnings.append(
                f".env file not found. Copy {env_example} to .env and configure it."
            )
            return False
        
        # Load environment variables
        load_dotenv()
        
        # Check critical variables
        required_vars = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            self.warnings.append(f"Missing environment variables: {', '.join(missing)}")
            return False
        
        return True
    
    def validate_gcp_safeguards(self) -> bool:
        """Ensure GCP cost safeguards are enabled"""
        valid = True
        
        # Check if free tier mode is enabled
        if not self.config.get("gcp", {}).get("usage_policy", {}).get("free_tier_mode", False):
            self.errors.append("GCP free_tier_mode must be enabled!")
            valid = False
        
        # Check if manual enable is required
        if not self.config.get("gcp", {}).get("usage_policy", {}).get("require_manual_enable", False):
            self.errors.append("GCP require_manual_enable must be true!")
            valid = False
        
        # Check if BigQuery is disabled by default
        if self.config.get("gcp", {}).get("bigquery", {}).get("enabled", True):
            self.errors.append("BigQuery must be disabled by default (set enabled: false)!")
            valid = False
        
        # Check budget limits
        budget = self.config.get("gcp", {}).get("budget", {})
        if budget.get("max_spend_usd", 0) > 10:
            self.warnings.append(
                f"Budget max_spend_usd is ${budget['max_spend_usd']}. "
                f"Consider lowering to $10 to preserve credits."
            )
        
        return valid
    
    def validate_local_dev_profile(self) -> bool:
        """Ensure local development profile is properly configured"""
        valid = True
        
        profile = self.config.get("app", {}).get("profile")
        if profile != "local_dev":
            self.warnings.append(
                f"Current profile is '{profile}'. Should be 'local_dev' for initial setup."
            )
        
        # Check resource policy for local_dev
        local_policy = self.config.get("resource_policy", {}).get("profiles", {}).get("local_dev", {})
        
        if local_policy.get("use_local_qdrant") != True:
            self.errors.append("local_dev profile must use local Qdrant!")
            valid = False
        
        if local_policy.get("use_local_postgres") != True:
            self.errors.append("local_dev profile must use local PostgreSQL!")
            valid = False
        
        if local_policy.get("use_remote_llm") != False:
            self.errors.append("local_dev profile should use local Ollama, not remote LLM!")
            valid = False
        
        return valid
    
    def validate_agent_config(self) -> bool:
        """Validate agent configurations"""
        valid = True
        
        # For local development, adaptive retrieval and critic should be disabled initially
        if self.config.get("app", {}).get("profile") == "local_dev":
            if self.config.get("agents", {}).get("adaptive_retrieval", {}).get("enabled", False):
                self.warnings.append(
                    "adaptive_retrieval should be disabled initially (enable after baseline)"
                )
            
            if self.config.get("agents", {}).get("critic", {}).get("enabled", False):
                self.warnings.append(
                    "critic agent should be disabled initially (enable for RLAIF training)"
                )
        
        return valid
    
    def validate_docker_resources(self) -> bool:
        """Check Docker resource allocations"""
        valid = True
        
        resources = self.config.get("deployment", {}).get("resources", {})
        
        # Ensure resources are reasonable for local development
        total_memory_gb = 0
        for service, limits in resources.items():
            if "memory" in limits:
                mem_str = limits["memory"]
                gb = int(mem_str.replace("G", ""))
                total_memory_gb += gb
        
        if total_memory_gb > 8:
            self.warnings.append(
                f"Total Docker memory allocation is {total_memory_gb}GB. "
                f"May be too high for local machines with <16GB RAM."
            )
        
        return valid
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks"""
        if not self.load_config():
            return False, self.errors, self.warnings
        
        self.check_env_file()
        self.validate_gcp_safeguards()
        self.validate_local_dev_profile()
        self.validate_agent_config()
        self.validate_docker_resources()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def print_report(self):
        """Print validation report"""
        is_valid, errors, warnings = self.validate_all()
        
        print("\n" + "="*70)
        print("  PatentSphere Configuration Validation Report")
        print("="*70)
        
        if errors:
            print("\n❌ ERRORS (must fix):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        if warnings:
            print("\n⚠️  WARNINGS (should review):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        if not errors and not warnings:
            print("\n✅ All checks passed! Configuration is valid.")
        
        print("\n" + "="*70)
        
        if is_valid:
            print("✅ READY TO PROCEED")
            print("\nNext steps:")
            print("  1. Copy .env.example to .env and configure credentials")
            print("  2. Start Docker services: docker-compose up -d")
            print("  3. Install Ollama and pull models")
            print("  4. Begin Day 1 tasks")
        else:
            print("❌ CONFIGURATION NEEDS FIXES")
            print("\nPlease resolve errors before proceeding.")
        
        print("="*70 + "\n")
        
        return is_valid


def main():
    """Run configuration validation"""
    validator = ConfigValidator()
    is_valid = validator.print_report()
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
