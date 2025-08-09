"""
Windows Executable Creator for Stress Level Prediction App
Creates a standalone .exe file using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("[OK] PyInstaller is already installed")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            print("[OK] PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install PyInstaller: {e}")
            return False

def create_spec_file():
    """Create a custom PyInstaller spec file for better control"""
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['stress_prediction_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dataset', 'dataset'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'pandas._libs.tslibs.timedeltas',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StressLevelPredictionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # Add icon path if you have one: icon='app_icon.ico'
)
"""
    
    with open('stress_app.spec', 'w') as f:
        f.write(spec_content.strip())
    print("[OK] Created custom spec file: stress_app.spec")

def create_executable_basic():
    """Create executable using basic PyInstaller command"""
    print("Creating executable with PyInstaller...")
    
    # Basic command - creates a single executable file
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Create single executable
        "--windowed",                   # No console window
        "--name", "StressLevelPredictionApp",
        "--add-data", "dataset;dataset",     # Include dataset folder
        "--add-data", "models;models",       # Include models folder
        "--hidden-import", "sklearn.utils._cython_blas",
        "--hidden-import", "sklearn.neighbors.typedefs", 
        "--hidden-import", "sklearn.neighbors.quad_tree",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "pandas._libs.tslibs.timedeltas",
        "stress_prediction_app.py"
    ]
    
    try:
        print("Running PyInstaller command...")
        print(" ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Executable created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyInstaller failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def create_executable_advanced():
    """Create executable using custom spec file"""
    print("Creating executable with custom spec file...")
    
    create_spec_file()
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "stress_app.spec"
    ]
    
    try:
        print("Running PyInstaller with spec file...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Executable created successfully with spec file!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyInstaller failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def create_installer_package():
    """Create a complete installer package"""
    print("Creating installer package...")
    
    # Create installer directory
    installer_dir = "WindowsInstaller"
    if os.path.exists(installer_dir):
        shutil.rmtree(installer_dir)
    os.makedirs(installer_dir)
    
    # Copy the executable
    if os.path.exists("dist/StressLevelPredictionApp.exe"):
        shutil.copy2("dist/StressLevelPredictionApp.exe", installer_dir)
        print("[OK] Copied executable to installer package")
    else:
        print("[ERROR] Executable not found in dist/")
        return False
    
    # Create installer batch script
    installer_script = """@echo off
echo ========================================
echo  Stress Level Prediction App Installer
echo ========================================
echo.

set INSTALL_DIR=%USERPROFILE%\\Desktop\\StressLevelApp

echo Creating installation directory...
mkdir "%INSTALL_DIR%" 2>nul

echo Copying application files...
copy "StressLevelPredictionApp.exe" "%INSTALL_DIR%\\"

echo Creating desktop shortcut...
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%USERPROFILE%\\Desktop\\Stress Level Prediction App.lnk" >> CreateShortcut.vbs  
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%INSTALL_DIR%\\StressLevelPredictionApp.exe" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "Stress Level Prediction App" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo ========================================
echo  Installation completed successfully!
echo ========================================
echo.
echo The application has been installed to:
echo %INSTALL_DIR%
echo.
echo A shortcut has been created on your desktop.
echo.
pause
"""
    
    with open(os.path.join(installer_dir, "install.bat"), 'w') as f:
        f.write(installer_script)
    
    print("[OK] Created installer script")
    return True

def cleanup_build_files():
    """Clean up temporary build files"""
    print("Cleaning up build files...")
    
    dirs_to_remove = ['build', '__pycache__']
    files_to_remove = ['stress_app.spec']
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"[OK] Removed {dir_name}")
    
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"[OK] Removed {file_name}")

def get_exe_size(exe_path):
    """Get executable file size in MB"""
    if os.path.exists(exe_path):
        size_bytes = os.path.getsize(exe_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0

def main():
    """Main executable creation function"""
    print("Windows Executable Creator for Stress Level Prediction App")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("stress_prediction_app.py"):
        print("[ERROR] stress_prediction_app.py not found!")
        print("Please run this script from the same directory as the main application.")
        return
    
    # Install PyInstaller
    if not install_pyinstaller():
        return
    
    # Ask user for creation method
    print("\nChoose executable creation method:")
    print("1. Basic (single file, faster)")
    print("2. Advanced (with spec file, more control)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    if not choice:
        choice = "1"
    
    success = False
    if choice == "2":
        success = create_executable_advanced()
    else:
        success = create_executable_basic()
    
    if success:
        exe_path = "dist/StressLevelPredictionApp.exe"
        if os.path.exists(exe_path):
            size_mb = get_exe_size(exe_path)
            print(f"\n[SUCCESS] Executable created!")
            print(f"Location: {os.path.abspath(exe_path)}")
            print(f"Size: {size_mb:.1f} MB")
            
            # Ask about creating installer package
            create_installer = input("\nCreate installer package? (y/N): ").strip().lower()
            if create_installer in ['y', 'yes']:
                if create_installer_package():
                    print("\n[SUCCESS] Installer package created in 'WindowsInstaller' folder")
            
            # Ask about cleanup
            cleanup = input("\nClean up build files? (Y/n): ").strip().lower()
            if cleanup not in ['n', 'no']:
                cleanup_build_files()
            
            print(f"\n{'='*60}")
            print("EXECUTABLE CREATION COMPLETED!")
            print(f"{'='*60}")
            print(f"📁 Executable: dist/StressLevelPredictionApp.exe")
            print(f"📦 Size: {size_mb:.1f} MB")
            print(f"🚀 Ready to distribute!")
            
        else:
            print("[ERROR] Executable was not created properly")
    else:
        print("[ERROR] Failed to create executable")

if __name__ == "__main__":
    main()
