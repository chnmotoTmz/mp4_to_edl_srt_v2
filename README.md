# MP4 to EDL/SRT Converter

このプロジェクトは、MP4動画ファイルから音声認識による字幕生成とシーン分析を行い、EDLファイルとSRT字幕ファイルを生成するツールです。

## 主な機能

1. **音声認識による字幕生成**
   - Whisperを使用した高精度な音声認識
   - 日本語対応
   - セグメントごとの文字起こしと評価

2. **シーン分析**
   - 動画のシーン分割
   - Gemini APIによるシーン説明文の生成
   - シーン評価タグの付与
   - サムネイル画像の生成

3. **出力フォーマット**
   - SRT字幕ファイルの生成
   - EDLファイル（CMX 3600形式）の生成
   - 詳細なJSON形式の出力データ

## システム要件

- Python 3.x
- FFmpeg
- CUDA対応GPU（推奨）

## 依存パッケージ

```
ffmpeg-python
faster-whisper>=0.9.0
pyyaml
whisper>=1.0.0
torch>=2.0.0
pydub>=0.25.1
opencv-python
```

## インストール方法

1. リポジトリをクローン：
```bash
git clone [リポジトリURL]
```

2. 依存パッケージをインストール：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

```bash
python main.py [動画ファイルのパス]
```

### オプション

- `--output_dir`: 出力ディレクトリの指定
- `--model_size`: Whisperモデルのサイズ指定（tiny, base, small, medium, large）
- `--device`: 処理デバイスの指定（cpu, cuda）
- `--language`: 音声認識の言語指定（デフォルト: ja）

## GUIの実行

- Windows: `run_gui.bat`を実行
- Linux/Mac: `run_gui.sh`を実行

## ファイル構造

```
.
├── main.py                 # メインスクリプト
├── scene_analysis.py       # シーン分析モジュール
├── transcription.py        # 音声認識モジュール
├── requirements.txt        # 依存パッケージリスト
├── run_gui.bat            # Windows用GUIランチャー
├── run_gui.sh             # Linux/Mac用GUIランチャー
└── config.json            # 設定ファイル
```

## 注意事項

1. **GPS機能について**
   - GPSデータの抽出機能は現在未実装です
   - 将来的な実装予定です

2. **処理時間について**
   - 音声認識とシーン分析には時間がかかります
   - GPUを使用することで処理時間を短縮できます

3. **リソース要件**
   - 大きな動画ファイルの処理には十分なメモリが必要です
   - GPUを使用する場合はCUDA対応のGPUが必要です

## エラー対処

1. 音声認識に失敗する場合：
   - 必要なパッケージが正しくインストールされているか確認
   - 音声トラックが正しく含まれているか確認
   - GPUメモリが不足していないか確認

2. シーン分析に失敗する場合：
   - 動画ファイルが正しい形式か確認
   - 十分なディスク容量があるか確認

## 出力データ仕様

### 1. GPSデータ（JSON形式）

```json
[
  {
    "timestamp": "2024-01-01T12:00:00",
    "latitude": 35.6895,
    "longitude": 139.6917,
    "altitude": 10.5,
    "speed": 5.2,
    "track": 90.0
  }
]
```

各フィールドの説明：
- `timestamp`: GPSデータの取得時刻（ISO 8601形式）
- `latitude`: 緯度（度単位）
- `longitude`: 経度（度単位）
- `altitude`: 高度（メートル単位）
- `speed`: 速度（メートル/秒）
- `track`: 進行方向（度単位、0-360）

### 2. SRT字幕ファイル

```
1
00:00:00,000 --> 00:00:05,000
字幕テキスト1

2
00:00:05,000 --> 00:00:10,000
字幕テキスト2
```

フォーマット仕様：
- 字幕番号（連番）
- タイムコード（開始時間 --> 終了時間）
  - 形式：HH:MM:SS,mmm
  - ミリ秒は3桁で表示
- 字幕テキスト（1行または複数行）
- 空行で区切り

### 3. EDLファイル（CMX 3600形式）

```
TITLE: プロジェクト名
FCM: NON-DROP FRAME

001  AX       V     C        00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
* FROM CLIP NAME: クリップ名
* COMMENT: コメント
```

各フィールドの説明：
- `TITLE`: プロジェクト名
- `FCM`: フレームカウントモード
- イベント行の構成：
  - イベント番号（3桁）
  - リールID（2文字）
  - トラックタイプ（V=ビデオ、A=オーディオ）
  - トランジションタイプ
  - ソースタイムコード（開始/終了）
  - レコードタイムコード（開始/終了）
- メタデータ行：
  - クリップ名
  - コメント

### 4. テキスト形式GPSデータ

```
総データ数: 100

ポイント 1:
  緯度: 35.6895
  経度: 139.6917
  高度: 10.5m
  速度: 5.2m/s
  時刻: 2024-01-01T12:00:00

ポイント 2:
  ...
```

フォーマット仕様：
- 総データ数の表示
- 各ポイントの詳細情報
  - 緯度・経度（度単位）
  - 高度（メートル単位）
  - 速度（メートル/秒）
  - 時刻（ISO 8601形式）

### 5. 設定ファイル（config.json）

```json
{
  "gps": {
    "output_format": "json",
    "include_metadata": true
  },
  "subtitle": {
    "language": "ja",
    "model_size": "medium",
    "use_gpu": true
  },
  "edl": {
    "frame_rate": 29.97,
    "timecode_format": "non-drop"
  }
}
```

設定項目：
- GPS設定
  - 出力フォーマット
  - メタデータの包含
- 字幕設定
  - 言語
  - モデルサイズ
  - GPU使用
- EDL設定
  - フレームレート
  - タイムコード形式

### 6. シーン分析データ（JSON形式）

```json
{
  "scenes": [
    {
      "scene_id": 1,
      "start_ms": 0,
      "end_ms": 5000,
      "description": "シーンの説明文",
      "thumbnail_path": "path/to/thumbnail.jpg",
      "scene_good_reason": "Scenic",
      "scene_bad_reason": null,
      "scene_evaluation_tag": "Scenic"
    }
  ]
}
```

各フィールドの説明：
- `scene_id`: シーンの連番ID
- `start_ms`: シーン開始時間（ミリ秒）
- `end_ms`: シーン終了時間（ミリ秒）
- `description`: Gemini APIによるシーン説明文
- `thumbnail_path`: シーンサムネイル画像のパス
- `scene_good_reason`: シーン評価（良い理由）
  - 有効な値: "Scenic", "Landmark", "Informative", "Action"
- `scene_bad_reason`: シーン評価（悪い理由）
  - 有効な値: "Privacy", "PoorQuality", "Irrelevant"
- `scene_evaluation_tag`: Gemini APIからの生の評価タグ

### 7. シーン評価タグ仕様

#### 良い評価タグ
- `Scenic`: 風景が美しいシーン
- `Landmark`: 有名な場所や建物が映っているシーン
- `Informative`: 情報量が多いシーン
- `Action`: アクションや動きのあるシーン

#### 悪い評価タグ
- `Privacy`: プライバシーに関わる可能性のあるシーン
- `PoorQuality`: 画質が悪いシーン
- `Irrelevant`: コンテンツと関係のないシーン

#### その他
- `Generic`: 特に特徴のない一般的なシーン
- `null`: 評価が行われていないシーン

### 8. シーン分析の設定パラメータ

```json
{
  "scene_analysis": {
    "frame_analysis_rate": 30,
    "capture_output_dir": "path/to/thumbnails",
    "model_path": "path/to/model",
    "device": "cpu"
  }
}
```

設定項目：
- `frame_analysis_rate`: 分析するフレームの間隔（Nフレームごと）
- `capture_output_dir`: サムネイル画像の保存先ディレクトリ
- `model_path`: シーン分析モデルのパス
- `device`: 処理デバイス（"cpu"または"cuda"）

### 9. 最終出力データ（JSON形式）

```json
{
    "source_filepath": "動画ファイルのパス",
    "file_index": 1,
    "extracted_audio_filepath": "抽出された音声ファイルのパス",
    "metadata": {
        "duration_seconds": 221.221,
        "creation_time_utc": "2016-01-05T21:14:57.000000Z",
        "timecode_offset": "01:21:49:54"
    },
    "transcription_whisper_result": {
        "text": "全文の文字起こしテキスト",
        "segments": [
            {
                "start": 3.88,
                "end": 4.66,
                "text": "セグメントの文字起こし"
            }
        ],
        "language": "ja"
    },
    "detected_scenes": [
        {
            "scene_id": 1,
            "start_timecode": "00:00:00:00",
            "end_timecode": "00:00:22:01",
            "description": "シーンの説明文",
            "thumbnail_path": "サムネイル画像のパス",
            "scene_good_reason": null,
            "scene_bad_reason": null,
            "scene_evaluation_tag": "Generic"
        }
    ],
    "final_segments": [
        {
            "start_timecode": "00:00:03:52",
            "end_timecode": "00:00:04:39",
            "transcription": "文字起こしテキスト",
            "scene_id": 1,
            "scene_description": "シーンの説明文",
            "transcription_good_reason": "Descriptive",
            "transcription_bad_reason": null,
            "source_timecode_offset": "01:21:49:54",
            "source_filename": "GH012562.MP4",
            "file_index": 1
        }
    ]
}
```

各フィールドの説明：

#### 基本情報
- `source_filepath`: 処理対象の動画ファイルパス
- `file_index`: ファイルのインデックス番号
- `extracted_audio_filepath`: 抽出された音声ファイルのパス

#### メタデータ
- `metadata.duration_seconds`: 動画の長さ（秒）
- `metadata.creation_time_utc`: 動画の作成時刻（UTC）
- `metadata.timecode_offset`: タイムコードのオフセット

#### 文字起こし結果
- `transcription_whisper_result.text`: 全文の文字起こし
- `transcription_whisper_result.segments`: セグメントごとの文字起こし
  - `start`: 開始時間（秒）
  - `end`: 終了時間（秒）
  - `text`: セグメントの文字起こし
- `transcription_whisper_result.language`: 検出された言語

#### シーン情報
- `detected_scenes`: 検出されたシーンのリスト
  - `scene_id`: シーンID
  - `start_timecode`: シーン開始タイムコード
  - `end_timecode`: シーン終了タイムコード
  - `description`: シーンの説明文
  - `thumbnail_path`: サムネイル画像のパス
  - `scene_good_reason`: 良い評価理由
  - `scene_bad_reason`: 悪い評価理由
  - `scene_evaluation_tag`: 評価タグ

#### 最終セグメント
- `final_segments`: 最終的なセグメントリスト
  - `start_timecode`: 開始タイムコード
  - `end_timecode`: 終了タイムコード
  - `transcription`: 文字起こしテキスト
  - `scene_id`: 対応するシーンID
  - `scene_description`: シーンの説明文
  - `transcription_good_reason`: 文字起こしの良い評価理由
    - 有効な値: "Descriptive", "Emotional"
  - `transcription_bad_reason`: 文字起こしの悪い評価理由
    - 有効な値: "Filler", "Repetitive"
  - `source_timecode_offset`: ソースのタイムコードオフセット
  - `source_filename`: ソースファイル名
  - `file_index`: ファイルインデックス

## ライセンス

このプロジェクトはオープンソースソフトウェアとして提供されています。

## 貢献

バグ報告や機能改善の提案は、Issueトラッカーを通じてお願いします。
