#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Stock Analyzer
Supports both GUI and CLI modes
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_gui():
    """Run the Streamlit GUI application"""
    try:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "src/gui.py", "--server.port=8501", "--server.address=localhost"]
        sys.exit(stcli.main())
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

def run_cli(ticker):
    """Run the CLI workflow"""
    try:
        from src.workflows.investment_workflow import run_investment_workflow
        run_investment_workflow(ticker)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--gui", "--streamlit", "-g"]:
            print("🚀 Starting GUI application...")
            run_gui()
        elif sys.argv[1] in ["--help", "-h"]:
            print("""
🤖 Multi-Agent Stock Analyzer

Usage:
  python main.py                    # Run GUI
  python main.py --gui             # Run GUI
  python main.py --streamlit       # Run GUI
  python main.py AAPL              # Run CLI analysis for AAPL
  python main.py --help            # Show this help

Examples:
  python main.py                   # Start the web interface
  python main.py TSLA              # Analyze TSLA via CLI
  python main.py --gui             # Start the web interface
            """)
        else:
            ticker = sys.argv[1].upper()
            print(f"📊 Analyzing {ticker}...")
            run_cli(ticker)
    else:
        print("🚀 Starting GUI application...")
        print("💡 Tip: Use 'python main.py --help' for more options")
        run_gui()

if __name__ == "__main__":
    main() 