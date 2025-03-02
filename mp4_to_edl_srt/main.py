import os
import argparse
import re
import subprocess
import glob
from typing import List, Dict

from mp4_file import MP4File
from edl_data import EDLData
from srt_data import SRTData
from segment import Segment


def process_folder(folder_path: str, output_folder: str = "output") -> None:
    """
    Processes a folder of MP4 files to generate combined EDL and SRT files.

    Args:
        folder_path: The path to the folder containing the MP4 files.
        output_folder: The folder to save the output EDL and SRT files.
    """
    # パスの正規化
    folder_path = os.path.normpath(folder_path)
    output_folder = os.path.normpath(output_folder)

    print(f"入力フォルダ: {folder_path}")
    print(f"出力フォルダ: {output_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"出力フォルダを作成しました: {output_folder}")

    edl_data = EDLData(title="My Video Project", fcm="NON-DROP FRAME")
    srt_data = SRTData()
    record_start = "00:00:00:00"

    # MP4ファイルの検索方法を改善
    mp4_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    if not mp4_files:
        print(f"警告: 指定されたフォルダ '{folder_path}' にMP4ファイルが見つかりません。")
        return

    print(f"MP4ファイル数: {len(mp4_files)}")
    for file in mp4_files:
        print(f" - {os.path.basename(file)}")

    for i, filepath in enumerate(mp4_files, start=1):
        filename = os.path.basename(filepath)
        try:
            print(f"処理中: {filename} ({i}/{len(mp4_files)})")
            mp4_file = MP4File(filepath, i)
            print(f"音声を抽出中...")
            mp4_file.extract_audio()
            print(f"文字起こし中...")
            mp4_file.transcribe()
            print(f"セグメント化中...")
            mp4_file.segment_audio(threshold=0.5)

            print(f"EDLデータを生成中...")
            file_edl, new_record_start = mp4_file.generate_edl_data(record_start)
            print(f"SRTデータを生成中...")
            file_srt = mp4_file.generate_srt_data()

            for event in file_edl.events:
                edl_data.add_event(event)
            
            for segment in file_srt.segments:
                srt_data.segments.append(segment)
                
            record_start = new_record_start

            # 一時ファイルのクリーンアップ
            if os.path.exists(mp4_file.audio_filepath):
                os.remove(mp4_file.audio_filepath)
                print(f"一時ファイルを削除しました: {mp4_file.audio_filepath}")

        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません: {filepath}")
            print(f"詳細: {str(e)}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg実行中にエラーが発生しました: {e}")
            print(f"詳細: {e.stderr if hasattr(e, 'stderr') else ''}")
        except Exception as e:
            print(f"{filename}の処理中に予期しないエラーが発生しました: {e}")
            import traceback
            print(traceback.format_exc())

    # 出力ファイルの書き込み
    output_edl_path = os.path.join(output_folder, "output.edl")
    output_srt_path = os.path.join(output_folder, "output.srt")
    
    try:
        with open(output_edl_path, "w", encoding="utf-8") as f:
            f.write(str(edl_data))
        
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(str(srt_data))
            
        print(f"生成されたファイル:")
        print(f" - EDL: {output_edl_path}")
        print(f" - SRT: {output_srt_path}")
    except Exception as e:
        print(f"出力ファイルの書き込み中にエラーが発生しました: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files to EDL and SRT.")
    parser.add_argument("--input", default="input_mp4_files", help="Input folder with MP4 files")
    parser.add_argument("--output", default="output", help="Output folder for EDL and SRT")
    args = parser.parse_args()
    process_folder(args.input, args.output)
