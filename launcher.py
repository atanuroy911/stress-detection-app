"""
Quick Launcher for Stress Level Prediction App
Simple script to start the application with error handling
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def launch_app():
    """Launch the main application"""
    try:
        # Check if main app exists
        if not os.path.exists("stress_prediction_app.py"):
            messagebox.showerror(
                "Error", 
                "stress_prediction_app.py not found!\n"
                "Please ensure all files are in the same directory."
            )
            return False
        
        # Check dependencies
        missing = check_dependencies()
        if missing:
            response = messagebox.askyesno(
                "Missing Dependencies",
                f"The following packages need to be installed:\n"
                f"{', '.join(missing)}\n\n"
                f"Install them now?"
            )
            if response:
                if not install_dependencies(missing):
                    messagebox.showerror(
                        "Installation Failed",
                        "Could not install required packages.\n"
                        "Please run: pip install -r requirements.txt"
                    )
                    return False
            else:
                return False
        
        # Launch main application
        subprocess.Popen([sys.executable, "stress_prediction_app.py"])
        return True
        
    except Exception as e:
        messagebox.showerror("Launch Error", f"Failed to launch application:\n{str(e)}")
        return False

def main():
    """Main launcher function"""
    # Hide the root window
    root = tk.Tk()
    root.withdraw()
    
    print("🚀 Launching Stress Level Prediction App...")
    
    if launch_app():
        print("✅ Application launched successfully!")
    else:
        print("❌ Failed to launch application")
        messagebox.showinfo(
            "Launch Failed", 
            "The application could not be started.\n"
            "Please check the console for error details."
        )

if __name__ == "__main__":
    main()
