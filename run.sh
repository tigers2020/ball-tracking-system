#!/bin/bash
# Run script for the Tennis Ball Tracker application
# This script activates the virtual environment and runs the main.py file

# Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo -e "\033[31mVirtual environment not found at .venv/bin/activate\033[0m"
    echo -e "\033[33mCreating a new virtual environment...\033[0m"
    python3 -m venv .venv
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo -e "\033[34mInstalling dependencies from requirements.txt...\033[0m"
        pip install -r requirements.txt
    else
        echo -e "\033[31mFailed to create virtual environment. Please check your Python installation.\033[0m"
        exit 1
    fi
fi

# Run the application
echo -e "\033[32mRunning Tennis Ball Tracker...\033[0m"
python main.py "$@" 