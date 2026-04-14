#!/bin/bash
# ──────────────────────────────────────────────
# NLC Classification Pipeline — Quick Launcher
# NASA Citizen Science — Space Cloud Watch
# ──────────────────────────────────────────────

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

echo ""
echo "=========================================="
echo "  NLC Classification Pipeline Launcher"
echo "=========================================="
echo ""
echo "  [1] Terminal  — Interactive Image Navigator"
echo "  [2] Portal    — Web Dashboard (Flask)"
echo ""
read -p "  Choose an option (1 or 2): " choice

echo ""

case $choice in
    1)
        echo ">> Running: python test_image_navigator.py"
        echo ""
        python "$SCRIPT_DIR/test_image_navigator.py"
        ;;
    2)
        echo ">> Running: python main.py portal --port 5001"
        echo ""
        python "$SCRIPT_DIR/main.py" portal --port 5001
        ;;
    *)
        echo "Invalid option. Please enter 1 or 2."
        exit 1
        ;;
esac
