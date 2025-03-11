# MP4 to EDL/SRT Converter

[English](#english) | [日本語](#japanese)

<a id="english"></a>
## English

### Overview
MP4 to EDL/SRT Converter is a tool that converts MP4 video files into EDL (Edit Decision List) and SRT subtitle files. It uses Whisper for speech recognition and supports both standard and high-speed modes.

### Features
- Converts MP4 files to EDL and SRT formats
- Supports batch processing of multiple MP4 files
- Uses OpenAI's Whisper for accurate speech recognition
- Supports faster-whisper for high-speed processing
- Preserves original timecodes from MP4 files
- User-friendly GUI interface
- Configurable settings via JSON

### Requirements
- Python 3.8 or later
- FFmpeg installed and accessible from PATH
- Required Python packages:
  ```
  whisper
  torch
  pydub
  faster-whisper (optional, for high-speed mode)
  ```

### Installation
1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install faster-whisper for high-speed mode:
   ```bash
   pip install faster-whisper
   ```

### Usage
#### GUI Mode
1. Run the GUI application:
   - Windows: Double-click `run_gui.bat`
   - Mac/Linux: Execute `run_gui.sh`
2. Select input folder containing MP4 files
3. Choose output folder for EDL/SRT files
4. Click "Start Conversion"

#### Command Line Mode
```bash
python main.py --input /path/to/input/folder --output /path/to/output/folder
```

Options:
- `--input`: Input folder containing MP4 files (required)
- `--output`: Output folder for EDL and SRT files (required)
- `--use-timecode`: Use MP4 file's internal timecode (default: True)
- `--no-timecode`: Ignore MP4 file's internal timecode

### Configuration
Settings can be customized in `config.json`:
- Whisper parameters (model, language, etc.)
- Audio processing settings
- Segmentation parameters
- GUI preferences

### Output Files
- `output.edl`: Edit Decision List in CMX 3600 format
- `output.srt`: Subtitle file with synchronized timecodes

<a id="japanese"></a>
## 日本語

### 概要
MP4 to EDL/SRT Converterは、MP4動画ファイルをEDL（編集決定リスト）とSRT字幕ファイルに変換するツールです。音声認識にWhisperを使用し、標準モードと高速モードの両方をサポートしています。

### 特徴
- MP4ファイルをEDLとSRT形式に変換
- 複数のMP4ファイルの一括処理に対応
- OpenAIのWhisperを使用した高精度な音声認識
- faster-whisperによる高速処理モードをサポート
- MP4ファイルの内部タイムコードを保持
- 使いやすいGUIインターフェース
- JSON形式での設定カスタマイズ

### 必要条件
- Python 3.8以降
- FFmpeg（PATHから実行可能な状態）
- 必要なPythonパッケージ：
  ```
  whisper
  torch
  pydub
  faster-whisper（オプション、高速モード用）
  ```

### インストール方法
1. このリポジトリをクローンまたはダウンロード
2. 必要なパッケージをインストール：
   ```bash
   pip install -r requirements.txt
   ```
3. （オプション）高速モード用にfaster-whisperをインストール：
   ```bash
   pip install faster-whisper
   ```

### 使用方法
#### GUIモード
1. GUIアプリケーションを起動：
   - Windows: `run_gui.bat`をダブルクリック
   - Mac/Linux: `run_gui.sh`を実行
2. MP4ファイルが入った入力フォルダを選択
3. EDL/SRTファイルの出力先フォルダを選択
4. 「変換開始」をクリック

#### コマンドラインモード
```bash
python main.py --input /path/to/input/folder --output /path/to/output/folder
```

オプション：
- `--input`: MP4ファイルが入ったフォルダ（必須）
- `--output`: EDLとSRTファイルの出力先フォルダ（必須）
- `--use-timecode`: MP4ファイルの内部タイムコードを使用（デフォルト：True）
- `--no-timecode`: 内部タイムコードを無視

### 設定
`config.json`で以下の設定をカスタマイズ可能：
- Whisperのパラメータ（モデル、言語など）
- 音声処理の設定
- セグメント化のパラメータ
- GUI表示設定

### 出力ファイル
- `output.edl`: CMX 3600形式の編集決定リスト
- `output.srt`: タイムコード同期済みの字幕ファイル

### 注意事項
- 処理時間は動画の長さやPCのスペックによって変動します
- 高速モードを使用する場合は、別途faster-whisperのインストールが必要です
- 大量のファイルを処理する場合は、十分なディスク容量を確保してください
