#!/bin/bash
# Quick Start Script for LumosTrade Terminal
# Run this to see what the terminal can do!

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🌟 LUMOSTRADE TERMINAL - QUICK START 🌟              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Choose an option:"
echo ""
echo "  1. Run the Terminal (Interactive)"
echo "  2. Run the Test Suite"
echo "  3. Run the Demo (Automated)"
echo "  4. View Documentation"
echo "  5. Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Starting LumosTrade Terminal..."
        echo "Type /help for commands, /exit to quit"
        echo ""
        python terminal/main.py
        ;;
    2)
        echo ""
        echo "Running Test Suite..."
        echo ""
        python terminal/test_terminal.py
        ;;
    3)
        echo ""
        echo "Running Demonstration..."
        echo ""
        python terminal/demo.py
        ;;
    4)
        echo ""
        echo "Opening README..."
        echo ""
        if command -v bat &> /dev/null; then
            bat terminal/README.md
        elif command -v less &> /dev/null; then
            less terminal/README.md
        else
            cat terminal/README.md
        fi
        ;;
    5)
        echo ""
        echo "Goodbye! 👋"
        echo ""
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid choice. Please run again and select 1-5."
        echo ""
        exit 1
        ;;
esac
