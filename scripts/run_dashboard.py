#!/usr/bin/env python3
"""
Faraday Dashboard Launcher

Launch the new Faraday dashboard with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Faraday dashboard."""
    # Set up environment
    dashboard_path = Path(__file__).parent.parent / "faraday" / "dashboard" / "main.py"

    if not dashboard_path.exists():
        print("❌ Dashboard main.py not found!")
        print(f"Expected path: {dashboard_path}")
        sys.exit(1)

    # Set Streamlit configuration
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"

    print("🚀 Starting Faraday Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 60)

    try:
        # Launch Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.headless",
            "true",
            "--server.address",
            "localhost",
            "--server.port",
            "8501",
            "--theme.base",
            "light",
            "--theme.primaryColor",
            "#1f4e79",
            "--theme.backgroundColor",
            "#ffffff",
            "--theme.secondaryBackgroundColor",
            "#f0f2f6",
        ]

        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
