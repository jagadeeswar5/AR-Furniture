#!/usr/bin/env python3
"""
AR Furniture App Startup Script
This script helps you start both the backend and frontend servers.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print a nice banner for the app"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🏠 AR Furniture App                      ║
    ║              Smart Interior Design Assistant                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if required files exist"""
    backend_dir = Path("backend")
    frontend_dir = Path("frontend")
    
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    if not (backend_dir / "main.py").exists():
        print("❌ Backend main.py not found!")
        return False
    
    print("✅ All required files found!")
    return True

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting Backend Server...")
    print("📍 Backend will run on: http://127.0.0.1:8000")
    
    backend_dir = Path("backend")
    os.chdir(backend_dir)
    
    try:
        # Start the backend server
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend server started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the frontend server"""
    print("\n🌐 Starting Frontend Server...")
    print("📍 Frontend will be available at: http://127.0.0.1:3000")
    
    # Go back to root directory
    os.chdir(Path(__file__).parent)
    frontend_dir = Path("frontend")
    
    try:
        # Try to use Python's built-in HTTP server
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", "3000"
        ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ Frontend server started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def open_browser():
    """Open the app in the default browser"""
    print("\n🌐 Opening app in your default browser...")
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:3000")

def main():
    """Main function to start the application"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please make sure you're in the correct directory and all files are present.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎯 Starting AR Furniture App...")
    print("="*60)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend. Please check the error messages above.")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend. Please check the error messages above.")
        backend_process.terminate()
        sys.exit(1)
    
    # Open browser
    open_browser()
    
    print("\n" + "="*60)
    print("🎉 AR Furniture App is now running!")
    print("="*60)
    print("📍 Frontend: http://127.0.0.1:3000")
    print("📍 Backend API: http://127.0.0.1:8000")
    print("📍 API Docs: http://127.0.0.1:8000/docs")
    print("\n💡 Tips:")
    print("   • Upload a room photo to get AI furniture recommendations")
    print("   • Use the drawing tools to mark furniture areas")
    print("   • Try the AR viewer to see furniture in 3D")
    print("   • Chat with the AI assistant for personalized advice")
    print("\n⚠️  Press Ctrl+C to stop both servers")
    print("="*60)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Servers stopped successfully!")
        print("👋 Thanks for using AR Furniture App!")

if __name__ == "__main__":
    main()


