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
- Settings configurable directly in the GUI
- **Bilingual interface (English/Japanese)**

### Requirements
- Python 3.8 or later
- FFmpeg installed and accessible from PATH
- Required Python packages:
  ```
  whisper>=1.0.0
  torch>=2.0.0
  pydub>=0.25.1
  faster-whisper>=0.9.0 (optional, for high-speed mode)
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
4. Configure optional settings in the GUI
5. **Select your preferred language (English/Japanese) from the interface**
6. Click "Start Conversion"

#### Command Line Mode
```bash
python main.py --input /path/to/input/folder --output /path/to/output/folder
```

Options:
- `--input`: Input folder containing MP4 files (required)
- `--output`: Output folder for EDL and SRT files (required)
- `--use-timecode`: Use MP4 file's internal timecode (default: True)
- `--no-timecode`: Ignore MP4 file's internal timecode

### Output Files
- `output.edl`: Edit Decision List in CMX 3600 format
- `output.srt`: Subtitle file with synchronized timecodes

## Development Status (Beta Version)

This project is currently in beta. Basic features are implemented, but there are several known issues.

### Implemented Features

- Audio extraction from MP4 files using FFmpeg
- Text transcription using Whisper AI
- EDL/SRT file generation
- Basic GUI interface with customizable settings
- **Bilingual interface (English/Japanese)**

### Known Issues

#### Audio Extraction Issues
- Some MP4 files may fail to detect audio streams
- Transcription accuracy varies significantly depending on audio quality

#### EDL/SRT Generation Challenges
- Timecode synchronization issues in some cases
- Increased memory usage when processing long videos

#### GUI Related
- Unstable progress bar updates
- No cancel functionality during processing

### Future Development Plans

- Enhanced Whisper model selection
- Expanded language setting options
- Audio processing parameter adjustment
- Memory usage optimization
- GUI functionality improvements and stabilization
- **Additional interface languages**

### Feedback Request

This tool is under development and has room for improvement. We appreciate feedback on:
- Bugs or operational issues
- Needed features or options
- Usage observations

### Handling Offline Clips in DaVinci Resolve

If you encounter offline media errors when importing EDL files into DaVinci Resolve, you can configure Resolve to continue rendering by:

1. Go to Preferences → User → User Interface Settings
2. Uncheck "Stop rendering when a frame or clip cannot be processed"
3. Click "Save"

This setting allows Resolve to skip offline clips during rendering rather than stopping the entire process.

## Settings Saving Feature

The application automatically saves the following settings and restores them on the next startup:

- Selected language (Japanese/English)
- Last used input folder path
- Last used output folder path

These settings are saved in a `settings.json` file. This file is typically created in the same directory as the application.

### Example of settings.json

```json
{
    "language": "en",
    "last_input_folder": "C:/Users/username/Videos",
    "last_output_folder": "C:/Users/username/Projects/edl_files"
}
```

For Japanese selection:

```json
{
    "language": "ja",
    "last_input_folder": "C:/Users/username/Documents/mp4_files",
    "last_output_folder": "C:/Users/username/Documents/edl_output"
}
```

You don't need to manually edit the settings file. Settings are automatically saved while using the application.

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
- GUI上で直接設定可能
- **バイリンガルインターフェース（日本語/英語）**

### 必要条件
- Python 3.8以降
- FFmpeg（PATHから実行可能な状態）
- 必要なPythonパッケージ：
  ```
  whisper>=1.0.0
  torch>=2.0.0
  pydub>=0.25.1
  faster-whisper>=0.9.0（オプション、高速モード用）
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
4. GUI上でオプション設定を行う
5. **インターフェースから希望の言語（日本語/英語）を選択**
6. 「変換開始」をクリック

#### コマンドラインモード
```bash
python main.py --input /path/to/input/folder --output /path/to/output/folder
```

オプション：
- `--input`: MP4ファイルが入ったフォルダ（必須）
- `--output`: EDLとSRTファイルの出力先フォルダ（必須）
- `--use-timecode`: MP4ファイルの内部タイムコードを使用（デフォルト：True）
- `--no-timecode`: 内部タイムコードを無視

### 出力ファイル
- `output.edl`: CMX 3600形式の編集決定リスト
- `output.srt`: タイムコード同期済みの字幕ファイル

## 開発状況（ベータ版）

本プロジェクトは現在ベータ版として開発中です。基本機能は実装されていますが、いくつかの既知の問題があります。

### 実装済みの基本機能

- FFmpegを使用したMP4からの音声抽出
- Whisper AIによる基本的な文字起こし
- EDL/SRTファイルの生成
- カスタマイズ可能な設定を備えた基本的なGUIインターフェース
- **バイリンガルインターフェース（日本語/英語）**

### 既知の問題点

#### 音声抽出に関する問題
- 一部のMP4ファイルで音声ストリームの検出に失敗することがある
- 音声品質によって文字起こしの精度が大きく変動する

#### EDL/SRT生成の課題
- タイムコードが正確に同期しないケースがある
- 長時間の動画処理時にメモリ使用量が増大する

#### GUI関連
- プログレスバーの更新が不安定
- 処理中のキャンセル機能が未実装

### 今後の開発予定

- Whisperモデルの選択機能の強化
- 言語設定オプションの拡充
- 音声処理パラメータの調整機能
- メモリ使用量の最適化
- GUIの機能改善と安定化
- **インターフェース言語の追加**

### フィードバックのお願い

このツールは開発途中であり、多くの改善の余地があります。以下のような情報をいただけると助かります：
- バグや動作の問題点
- 必要な機能やオプション
- 使用時の気づき

### DaVinci Resolveでのオフラインクリップエラー対処法

EDLファイルをDaVinci Resolveにインポートした際にオフラインメディアエラーが発生した場合、以下の設定でレンダリングを中断せず続行できます:

1. 「環境設定」→「ユーザー」→「ユーザーインターフェース設定」を開く
2. 「処理できないフレーム/クリップがある場合にレンダリングを停止」のチェックを外す
3. 「保存」をクリック

この設定により、オフラインクリップはスキップされ、レンダリングプロセス全体が停止することなく完了します。

### 注意事項
- 処理時間は動画の長さやPCのスペックによって変動します
- 高速モードを使用する場合は、別途faster-whisperのインストールが必要です
- 大量のファイルを処理する場合は、十分なディスク容量を確保してください

## 設定保存機能について

アプリケーションは以下の設定を自動的に保存し、次回起動時に復元します：

- 選択された言語（日本語/英語）
- 最後に使用した入力フォルダのパス
- 最後に使用した出力フォルダのパス

これらの設定は `settings.json` ファイルに保存されます。このファイルは通常、アプリケーションと同じディレクトリに作成されます。

### settings.jsonの例

```json
{
    "language": "ja",
    "last_input_folder": "C:/Users/username/Documents/mp4_files",
    "last_output_folder": "C:/Users/username/Documents/edl_output"
}
```

英語を選択した場合：

```json
{
    "language": "en",
    "last_input_folder": "C:/Users/username/Videos",
    "last_output_folder": "C:/Users/username/Projects/edl_files"
}
```

手動で設定ファイルを編集する必要はありません。アプリケーションの使用中に設定が自動的に保存されます。
