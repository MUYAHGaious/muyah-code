"""
Global CLI Installer
====================
Installs 'ccode' and 'colab-code' commands into your Python Scripts directory
so you can run it from any terminal anywhere on your system.
"""

import sys
from pathlib import Path

target_dirs = [
    Path.home() / ".local" / "bin",
    Path(sys.prefix) / "Scripts",
    Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts",
    Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Scripts"
]

script_path = Path(__file__).parent.resolve() / "code_agent.py"
batch_content = f'@echo off\npython "{script_path}" %*\n'

installed = 0
for d in target_dirs:
    if d.exists():
        for name in ["ccode.cmd", "colab-code.cmd"]:
            target_file = d / name
            try:
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(batch_content)
                print(f"[+] Installed global command: {target_file}")
                installed += 1
            except Exception as e:
                print(f"[-] Could not write to {target_file}: {e}")

if installed > 0:
    print("\n Success! You can now open ANY terminal anywhere on your computer and type:")
    print("   ccode")
    print("   or")
    print("   colab-code")
