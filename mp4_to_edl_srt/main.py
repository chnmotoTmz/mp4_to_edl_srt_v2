import os
import argparse
import re
import subprocess
import glob
from typing import List, Dict, Tuple

from mp4_file import MP4File
from edl_data import EDLData
from srt_data import SRTData
from segment import Segment


def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Processes a folder of MP4 files to generate combined EDL and SRT files.
    
    Args:
        input_folder: Path to the folder containing MP4 files.
        output_folder: Path to save the output EDL and SRT files.
    """
    # 入力・出力フォルダのパスを正規化
    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(output_folder)
    
    print(f"入力フォルダ: {input_folder}")
    print(f"出力フォルダ: {output_folder}")
    
    # 環境変数から初期プロンプトを取得
    initial_prompt = os.environ.get("WHISPER_INITIAL_PROMPT", "日本語での自然な会話。文脈に応じて適切な表現を使用してください。")
    print(f"初期プロンプト: {initial_prompt}")
    
    # 環境変数からWhisperパラメータを取得
    temperature = os.environ.get("WHISPER_TEMPERATURE", "0.2")
    beam_size = os.environ.get("WHISPER_BEAM_SIZE", "5")
    condition_on_previous = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS", "True")
    enable_preprocessing = os.environ.get("ENABLE_AUDIO_PREPROCESSING", "True")
    
    print(f"Whisperパラメータ設定:")
    print(f" - Temperature: {temperature}")
    print(f" - Beam Size: {beam_size}")
    print(f" - 文脈考慮: {condition_on_previous}")
    print(f" - 音声前処理: {enable_preprocessing}")
    
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"出力フォルダを作成しました: {output_folder}")
    
    # MP4ファイルを検索
    mp4_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    if not mp4_files:
        print(f"警告: 指定されたフォルダ({input_folder})にMP4ファイルが見つかりません。")
        return
    
    print(f"{len(mp4_files)}個のMP4ファイルを処理します...")
    
    # EDLとSRTデータの初期化
    edl_data = EDLData(title="MP4 to EDL Project", fcm="NON-DROP FRAME")
    srt_data = SRTData()
    
    # 次のレコードの開始時間（最初は00:00:00:00から）
    next_record_start = "00:00:00:00"
    
    # 処理済みのMP4ファイルオブジェクトを保存するリスト
    processed_mp4_files = []
    
    # 各MP4ファイルを処理
    for i, mp4_file_path in enumerate(sorted(mp4_files), 1):
        try:
            print(f"ファイル {i}/{len(mp4_files)} を処理中: {os.path.basename(mp4_file_path)}")
            
            # MP4ファイルを処理
            mp4_file = MP4File(mp4_file_path, i)
            mp4_file.extract_audio()
            mp4_file.transcribe(initial_prompt=initial_prompt)  # 初期プロンプトを渡す
            mp4_file.segment_audio(threshold=0.5)
            
            # EDLデータを生成して結合
            file_edl_data, next_record_start = mp4_file.generate_edl_data(next_record_start)
            for event in file_edl_data.events:
                edl_data.add_event(event)
            
            # 処理済みのMP4ファイルオブジェクトを保存
            processed_mp4_files.append(mp4_file)
                
        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません: {e}")
            continue
        except subprocess.CalledProcessError as e:
            print(f"エラー: FFmpegの実行中にエラーが発生しました: {e}")
            continue
        except Exception as e:
            print(f"エラー: 処理中に予期しないエラーが発生しました: {e}")
            continue
    
    # EDLファイルを書き込み
    edl_output_path = os.path.join(output_folder, "output.edl")
    with open(edl_output_path, "w", encoding="utf-8") as f:
        f.write(str(edl_data))
    print(f"EDLファイルを保存しました: {edl_output_path}")
    
    # EDL生成後にSRTデータを生成
    print(f"EDLに基づいてSRTデータを生成します...")
    for mp4_file in processed_mp4_files:
        # EDLのレコードタイムコードに基づいてSRTデータを生成
        file_srt_data = mp4_file.generate_srt_data()
        for segment in file_srt_data.segments:
            srt_data.add_segment(segment)
    
    # SRTファイルを書き込み
    srt_output_path = os.path.join(output_folder, "output.srt")
    srt_data.write_to_file(srt_output_path)
    print(f"SRTファイルを保存しました: {srt_output_path}")
    print(f"EDLとSRTのタイムコードが同期されました")

def main():
    """Main function to parse arguments and process the folder."""
    parser = argparse.ArgumentParser(description="MP4 to EDL/SRT Converter")
    parser.add_argument("--input", required=True, help="Input folder containing MP4 files")
    parser.add_argument("--output", required=True, help="Output folder for EDL and SRT files")
    
    args = parser.parse_args()
    process_folder(args.input, args.output)

if __name__ == "__main__":
    main()
