import os
import re
import subprocess
from typing import List, Dict

import pydub
import whisper
from edl_data import EDLData
from mp4_file import MP4File
from srt_data import SRTData
from segment import Segment


def process_folder(folder_path: str, output_folder: str = "output") -> None:
    """
    Processes a folder of MP4 files, extracting audio, transcribing, segmenting,
    and generating EDL and SRT files.

    Args:
        folder_path: The path to the folder containing the MP4 files.
        output_folder: The folder to save the output EDL and SRT files.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            filepath = os.path.join(folder_path, filename)
            try:
                mp4_file = MP4File(filepath)
                mp4_file.extract_audio()
                mp4_file.transcribe()
                mp4_file.segment_audio(threshold=0.5)  # Default threshold

                edl_data = mp4_file.generate_edl_data()
                srt_data = mp4_file.generate_srt_data()

                # Write output files to the specified output folder
                output_edl_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.edl")
                output_srt_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.srt")

                with open(output_edl_path, "w") as f:
                    f.write(str(edl_data))  # Write as string for simplicity

                with open(output_srt_path, "w") as f:
                    f.write(str(srt_data))  # Write as string for simplicity

            except FileNotFoundError:
                print(f"Error: File not found: {filepath}")
            except subprocess.CalledProcessError as e:
                print(f"Error during FFmpeg execution: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


if __name__ == "__main__":
    input_folder = "input_mp4_files"  # Replace with your input folder
    process_folder(input_folder)
