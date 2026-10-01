#!/bin/bash
# Double-click this file (macOS) to start the backtester UI in your browser.
# It opens Terminal, starts the app, and opens http://localhost:8501 (or the next free port).
# Close the Terminal window (or press Ctrl-C in it) to stop the app.
cd "$(dirname "$0")" || exit 1
echo "Starting the Factor Ranking Backtester from: $(pwd)"
echo "First start takes ~10 seconds. Leave this window open while you use the app."
echo
python3 -m streamlit run app.py
