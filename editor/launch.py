#!/usr/bin/env python3
"""
Spec Editor - One-Click Launcher
Just run: python launch.py
"""

import sys
import time
import webbrowser
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'flask': 'flask',
        'flask_cors': 'flask-cors',
        'requests': 'requests',
        'dotenv': 'python-dotenv'
    }
    missing = []

    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies(packages):
    """Install missing packages."""
    import subprocess
    print(f"\n📦 Installing: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
        print("✓ Dependencies installed!\n")
        return True
    except subprocess.CalledProcessError:
        print("✗ Installation failed")
        return False

def main():
    print("\n" + "=" * 60)
    print("  🎨 Spec Editor")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("\n✗ Error: Python 3.8 or higher is required")
        print(f"  You have: Python {sys.version_info.major}.{sys.version_info.minor}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print(f"\n✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check and install dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}")
        response = input("\n   Install them now? (y/n): ").strip().lower()
        if response != 'y':
            print("\n✗ Cannot start without dependencies")
            input("\nPress Enter to exit...")
            sys.exit(1)

        if not install_dependencies(missing):
            print("\n✗ Please install manually:")
            print(f"   pip install {' '.join(missing)}")
            input("\nPress Enter to exit...")
            sys.exit(1)
    else:
        print("✓ All dependencies installed")

    # Now import Flask (after potential installation)
    try:
        from flask import Flask, send_from_directory, request, jsonify
        from flask_cors import CORS
        import requests as req
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\n   Try reinstalling:")
        print("   pip install flask flask-cors requests python-dotenv")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Import the app from server.py
    print("\n🚀 Starting server...")

    # Change to editor directory
    editor_dir = Path(__file__).parent
    os.chdir(editor_dir)

    # Import the Flask app
    try:
        from server import app
    except Exception as e:
        print(f"\n✗ Error loading server: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Server info
    PORT = 5001
    URL = f"http://localhost:{PORT}"

    print("\n" + "=" * 60)
    print("  ✓ Server Ready!")
    print("=" * 60)
    print(f"\n  📍 URL: {URL}")
    print(f"  📂 Directory: {editor_dir}")
    print("\n  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(URL)
            print("✓ Browser opened\n")
        except:
            print("⚠️  Could not open browser automatically")
            print(f"   Please open: {URL}\n")

    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Start the server
    try:
        app.run(host='localhost', port=PORT, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n⚠️  Port 5001 is already in use!")
            print("\n   Either:")
            print("   1. Open http://localhost:5001 (server may already be running)")
            print("   2. Stop the other server and try again")
            print("\n   On Mac/Linux, run: lsof -ti:5001 | xargs kill")
        else:
            print(f"\n✗ Server error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping server...")
        print("✓ Server stopped\n")
        sys.exit(0)

if __name__ == '__main__':
    main()
