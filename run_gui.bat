@echo off
chcp 65001 > nul
cd /d %~dp0
echo MP4 to EDL/SRT Converter を起動しています...
echo.

REM Pythonが利用可能か確認
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo エラー: Pythonが見つかりません。
    echo Pythonをインストールしてから再試行してください。
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 必要なパッケージがインストールされているか確認
python -c "import tkinter" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 警告: tkinterがインストールされていません。
    echo GUIが正常に動作しない可能性があります。
)

REM PYTHONPATHを設定
set PYTHONPATH=%~dp0

echo GUIアプリケーションを起動しています...
python mp4_to_edl_srt/gui.py
if errorlevel 1 (
    echo エラーが発生しました。詳細はログを確認してください。
    pause
)

pause 