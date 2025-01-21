import os
from pathlib import Path
import winreg
import platform
import re

def get_npcshrc_path():
    """Get the .npcshrc path regardless of platform"""
    return Path.home() / '.npcshrc'

def get_user_input(prompt, default=None):
    if default:
        response = input(f"{prompt} (default: {default}): ").strip()
        return response if response else default
    return input(f"{prompt}: ").strip()

def write_rc_file(path, config):
    """Write config in shell rc file format"""
    with open(path, 'w') as f:
        for key, value in config.items():
            # Ensure values are properly quoted
            quoted_value = f"'{value}'" if "'" not in str(value) else f'"{value}"'
            f.write(f"{key}={quoted_value}\n")

def read_rc_file(path):
    """Read shell-style rc file"""
    config = {}
    if not path.exists():
        return config
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Match KEY='value' or KEY="value" format
                match = re.match(r'^([A-Z_]+)\s*=\s*[\'"](.*?)[\'"]$', line)
                if match:
                    key, value = match.groups()
                    config[key] = value
    return config

def setup_windows_config():
    if platform.system() != "Windows":
        print("This setup script is for Windows only.")
        return

    npcshrc_path = get_npcshrc_path()

    print("\n=== NPCSH Windows Setup ===\n")
    print("This script will help you configure NPCSH on your Windows system.\n")

    # Get user preferences
    model = get_user_input("Enter your preferred AI model", "claude-3-sonnet")
    provider = get_user_input("Enter your preferred AI provider", "anthropic")
    openai_key = get_user_input("Enter your OpenAI API key (press Enter to skip)")
    anthropic_key = get_user_input("Enter your Anthropic API key (press Enter to skip)")

    # Create config
    config = {}
    if openai_key:
        config['OPENAI_API_KEY'] = openai_key
    if anthropic_key:
        config['ANTHROPIC_API_KEY'] = anthropic_key
    config['NPCSH_MODEL'] = model
    config['NPCSH_PROVIDER'] = provider

    # Read existing config if it exists
    if npcshrc_path.exists():
        existing_config = read_rc_file(npcshrc_path)
        # Update with new values, preserving any existing ones
        existing_config.update(config)
        config = existing_config

    # Save config in rc file format
    write_rc_file(npcshrc_path, config)

    # Set environment variables
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_ALL_ACCESS) as key:
            for var_name, var_value in config.items():
                winreg.SetValueEx(key, var_name, 0, winreg.REG_SZ, var_value)
    except WindowsError as e:
        print(f"\nWarning: Failed to set environment variables: {e}")
        print("You may need to set them manually.")

    # Add Scripts directory to PATH
    try:
        import site
        user_scripts = Path(site.USER_BASE) / "Scripts"
        current_path = os.environ.get('PATH', '')
        
        if str(user_scripts) not in current_path:
            os.system(f'setx PATH "%PATH%;{user_scripts}"')
    except Exception as e:
        print(f"\nWarning: Failed to update PATH: {e}")
        print(f"You may need to add {user_scripts} to your PATH manually.")

    print("\n=== Setup Complete ===")
    print(f"Configuration saved to: {npcshrc_path}")
    print("\nNotes:")
    print("1. You may need to restart your terminal for environment variables to take effect")
    print("2. To modify settings later, edit ~/.npcshrc or run this setup again")
    print("3. The 'npcsh' command should now be available in new terminal windows")

if __name__ == "__main__":
    setup_windows_config()