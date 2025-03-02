#!/bin/bash
export LANG=ja_JP.UTF-8

echo "MP4 to EDL/SRT Converter を起動しています..."
echo

# 現在のディレクトリに移動
cd "$(dirname "$0")"

# Pythonが利用可能か確認
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonが見つかりません。"
    echo "Pythonをインストールしてから再試行してください。"
    echo "https://www.python.org/downloads/"
    read -p "Enterキーを押して終了..."
    exit 1
fi

# 必要なパッケージがインストールされているか確認
if ! python -c "import tkinter" &> /dev/null; then
    echo "警告: tkinterがインストールされていません。"
    echo "GUIが正常に動作しない可能性があります。"
fi

echo "GUIアプリケーションを起動しています..."
python mp4_to_edl_srt/gui.py

if [ $? -ne 0 ]; then
    echo "エラーが発生しました。詳細はログを確認してください。"
fi

read -p "Enterキーを押して終了..." 