@echo off
chcp 65001 > nul
cd /d %~dp0
echo Launching MP4 to EDL/SRT Converter (Electron App)...
echo.

REM Check if Node.js and npm are available
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js is not found.
    echo Please install Node.js (which includes npm) and try again.
    echo https://nodejs.org/
    pause
    exit /b 1
)

npm --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm is not found.
    echo Please install Node.js (which includes npm) and try again.
    echo https://nodejs.org/
    pause
    exit /b 1
)

echo Starting the Electron application...
REM This assumes package.json has a "start": "electron ." script.
npm start

if %ERRORLEVEL% neq 0 (
    echo An error occurred while trying to start the Electron application.
    echo Please check the console for more details.
    pause
    exit /b 1
)

echo.
echo The Electron application has been started.
echo If the app window does not appear, please check for error messages above.
pause