#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script to validate MLA setup with LMCache.
"""

import sys
import yaml
from pathlib import Path


def check_imports():
    """Check if required packages are installed."""
    required_packages = {
        'lmcache': 'LMCache',
        'vllm': 'vLLM',
        'openai': 'OpenAI client',
        'transformers': 'Transformers',
        'requests': 'Requests'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing.append(package)
    
    return len(missing) == 0


def check_config():
    """Check if configuration file is valid."""
    config_file = Path("lmcache_config.yaml")
    
    if not config_file.exists():
        print("✗ Configuration file not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical MLA settings
        if config.get('remote_serde') != 'naive':
            print("⚠ Warning: MLA models require 'remote_serde: naive'")
        
        if config.get('use_layerwise', False):
            print("⚠ Warning: MLA models don't support layerwise operations")
        
        print("✓ Configuration file is valid")
        return True
    except Exception as e:
        print(f"✗ Configuration file error: {e}")
        return False


def check_scripts():
    """Check if example scripts are present and executable."""
    scripts = [
        ("mla_example.py", "Main example script"),
        ("launch_server.sh", "Server launch script")
    ]
    
    all_present = True
    for script, description in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"✓ {description} found: {script}")
        else:
            print(f"✗ {description} NOT found: {script}")
            all_present = False
    
    return all_present


def main():
    print("="*60)
    print("MLA Example Setup Validation")
    print("="*60)
    
    print("\n1. Checking Python packages...")
    packages_ok = check_imports()
    
    print("\n2. Checking configuration...")
    config_ok = check_config()
    
    print("\n3. Checking example scripts...")
    scripts_ok = check_scripts()
    
    print("\n" + "="*60)
    if packages_ok and config_ok and scripts_ok:
        print("✓ Setup is complete! You can now run the example.")
        print("\nNext steps:")
        print("1. Start the server: ./launch_server.sh")
        print("2. Run the example: python mla_example.py")
        return 0
    else:
        print("✗ Setup is incomplete. Please address the issues above.")
        if not packages_ok:
            print("\nInstall missing packages: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())