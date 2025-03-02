# MP4 to EDL/SRT Converter

MP4動画ファイルから音声を抽出し、Whisperを使用して音声を文字起こしし、EDL（Edit Decision List）とSRT（字幕）ファイルを生成するPythonアプリケーションです。

## 機能

- MP4ファイルから音声を抽出
- Whisperを使用した高精度な音声文字起こし
- 無音部分を検出して音声をセグメント化
- CMX 3600形式のEDLファイル生成
- SRT形式の字幕ファイル生成
- 使いやすいGUIインターフェース

## 必要条件

- Python 3.7以上
- FFmpeg（システムにインストールされていること）
- 以下のPythonパッケージ:
  - openai-whisper
  - pydub
  - ffmpeg-python
  - tkinter（通常はPythonに同梱）

## インストール

1. このリポジトリをクローンまたはダウンロードします。
2. 必要なパッケージをインストールします：

```bash
pip install openai-whisper pydub ffmpeg-python
```

3. FFmpegをインストールします（まだインストールしていない場合）：
   - Windows: [FFmpeg公式サイト](https://ffmpeg.org/download.html)からダウンロードし、PATHに追加します。
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`（Ubuntuの場合）

## 使い方

### GUIモード（推奨）

1. 以下のいずれかの方法でGUIを起動します：
   - Windows: `run_gui.bat`をダブルクリック
   - macOS/Linux: ターミナルで`./run_gui.sh`を実行（必要に応じて`chmod +x run_gui.sh`で実行権限を付与）
   - または、コマンドラインで`python mp4_to_edl_srt/gui.py`を実行

2. GUIが起動したら：
   - 「入力フォルダ」に処理したいMP4ファイルが含まれるフォルダを指定します。
   - 「出力フォルダ」にEDLとSRTファイルを保存するフォルダを指定します。
   - 「変換開始」ボタンをクリックして処理を開始します。
   - 処理の進行状況はログエリアに表示されます。

### コマンドラインモード

```bash
python mp4_to_edl_srt/main.py --input <入力フォルダ> --output <出力フォルダ>
```

## 出力ファイル

- `output.edl`: CMX 3600形式のEDLファイル（DaVinci Resolveなどの編集ソフトで使用可能）
- `output.srt`: SRT形式の字幕ファイル

## 注意事項

- 処理時間は入力ファイルのサイズと数、およびコンピュータの性能に依存します。
- Whisperの文字起こし精度は音声の品質に依存します。
- 30fps非ドロップフレームを前提にタイムコードを計算しています。

## トラブルシューティング

- FFmpegが見つからないエラーが表示される場合は、FFmpegが正しくインストールされ、PATHに追加されていることを確認してください。
- メモリエラーが発生する場合は、より小さなMP4ファイルで試すか、コンピュータのメモリを増設してください。
- その他の問題が発生した場合は、ログを確認して問題を特定してください。 "# mp4_to_edl_srt" 
