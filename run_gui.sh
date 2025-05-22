#!/bin/bash
# export LANG=ja_JP.UTF-8 # Keep if needed for specific locales, otherwise optional

echo "Launching MP4 to EDL/SRT Converter (Electron App)..."
echo

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

# Check if Node.js and npm are available (optional, but good for user feedback)
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not found."
    echo "Please install Node.js (which includes npm) and try again."
    echo "https://nodejs.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not found."
    echo "Please install Node.js (which includes npm) and try again."
    echo "https://nodejs.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting the Electron application..."
# This assumes package.json has a "start": "electron ." script.
npm start

if [ $? -ne 0 ]; then
    echo "An error occurred while trying to start the Electron application."
    echo "Please check the console for more details."
fi

echo
echo "The Electron application has been started."
echo "If the app window does not appear, please check for error messages above."
read -p "Press Enter to exit..."