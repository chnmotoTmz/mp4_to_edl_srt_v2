import os
import json
# import argparse # Removed argparse
import re
import logging
logger = logging.getLogger(__name__) # Define the logger

import subprocess
import glob
from typing import List, Dict, Tuple, Optional, Callable # Added Optional, Callable

from mp4_to_edl_srt.mp4_file import MP4File, ConfigManager # Import ConfigManager
from mp4_to_edl_srt.edl_data import EDLData
from mp4_to_edl_srt.srt_data import SRTData
from mp4_to_edl_srt.segment import Segment # Import Segment class for type hinting
from mp4_to_edl_srt.scene_analysis import SceneAnalyzer # Import SceneAnalyzer
from mp4_to_edl_srt.timecode_utils import TimecodeConverter # Import TimecodeConverter

def update_progress(value: float, stage: str):
    """Simple progress callback function (can be replaced by GUI update)."""
    # In a real GUI, this would update a progress bar or label.
    print(f"Progress: {stage} - {value*100:.1f}%")

# Global converter and config instances (can be managed differently, e.g., in GUI)
config_manager = ConfigManager()
converter = TimecodeConverter() # Use default frame rate (30) or load from config if added

def process_folder(input_folder: str, output_folder: str, progress_callback: Optional[Callable[[float, str], None]] = update_progress) -> None:
    """
    Processes a folder of MP4 files to generate combined EDL and SRT files.

    Args:
        input_folder: Path to the folder containing MP4 files.
        output_folder: Path to save the output EDL and SRT files.
        progress_callback: Function to report progress (value, stage_name).
    """
    # Load config settings
    config = config_manager.config
    whisper_config = config.get('whisper', {})
    edl_config = config.get('edl', {})
    scene_config = config.get('scene_analysis', {})
    use_timecode_offset = edl_config.get('use_timecode_offset', True)

    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(output_folder)

    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Using Internal Timecode Offset: {use_timecode_offset}")
    print(f"Whisper Model: {whisper_config.get('model', 'small')}, Lang: {whisper_config.get('language', 'ja')}")
    print(f"Scene Analysis Enabled: {scene_config.get('enabled', False)}, Rate: {scene_config.get('frame_analysis_rate', 30)}")

    if progress_callback:
        progress_callback(0.0, "Starting")

    # Initialize SceneAnalyzer if enabled (requires converter for EDL/SRT generation later)
    scene_analyzer_instance: Optional[SceneAnalyzer] = None
    if scene_config.get('enabled', False):
        # Define a wrapper here that translates the scene analyzer's float progress
        # back to the format expected by the main progress callback (value, stage)
        # This needs access to 'progress_callback', 'progress_start', 'progress_end', 'current_stage_base'
        def scene_analyzer_progress_wrapper(scene_prog: float):
            if progress_callback:
                # Map scene analysis progress (0-1) to the file's overall progress range (e.g., 70-90%)
                file_progress = 0.7 + scene_prog * 0.2
                overall_progress = progress_start + file_progress * (progress_end - progress_start)
                progress_callback(overall_progress, f"{current_stage_base} - Analyzing Scenes")

        # Initialize SceneAnalyzer (pass the converter)
        scene_analyzer_instance = SceneAnalyzer(
             # model_path=None, # Removed, handled internally
             # device='cpu', # Removed, handled internally
             progress_callback=scene_analyzer_progress_wrapper
        )
        print("Scene Analyzer Initialized.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")

    mp4_files = glob.glob(os.path.join(input_folder, "*.mp4"))

    if not mp4_files:
        print(f"Warning: No MP4 files found in {input_folder}.")
        if progress_callback:
            progress_callback(1.0, "No files found")
        return

    print(f"Processing {len(mp4_files)} MP4 files...")

    # Create single lists to hold all segments for final processing
    all_segments: List[Segment] = [] 

    total_files = len(mp4_files)
    files_processed_successfully = 0

    for i, mp4_file_path in enumerate(sorted(mp4_files), 1):
        file_basename = os.path.basename(mp4_file_path)
        current_stage_base = f"File {i}/{total_files} ({file_basename})"
        print(f"\n--- Processing {current_stage_base} ---")

        try:
            # Calculate progress range for this file (rough estimate)
            progress_start = (i - 1) / total_files
            progress_end = i / total_files

            def file_progress_callback(value: float, stage: str):
                 # Map file progress (0-1) to overall progress range
                 overall_progress = progress_start + value * (progress_end - progress_start)
                 if progress_callback:
                      progress_callback(overall_progress, f"{current_stage_base} - {stage}")

            if progress_callback:
                file_progress_callback(0.0, "Initializing")

            # Initialize MP4File with converter and scene analyzer
            mp4_file = MP4File(mp4_file_path, i, converter, scene_analyzer_instance)

            # 1. Extract Audio
            if progress_callback:
                file_progress_callback(0.05, "Extracting Audio")
            mp4_file.extract_audio()

            # 2. Transcribe Audio
            if progress_callback:
                file_progress_callback(0.1, "Transcribing")
            # Pass whisper config and a wrapped progress callback for transcription phase
            def transcription_progress_wrapper(step, total_steps):
                try:
                    # Check if step and total_steps are valid numbers
                    if isinstance(step, (int, float)) and isinstance(total_steps, (int, float)) and total_steps > 0:
                        file_progress = 0.1 + (step / total_steps) * 0.6
                    else:
                        # If steps are invalid or total is zero, show minimal progress
                        file_progress = 0.11 # Indicate slightly past start
                        logger.debug(f"Invalid progress data from Whisper callback: step={step}, total_steps={total_steps}")
                    file_progress_callback(file_progress, "Transcribing")
                except Exception as cb_err:
                    # Log any error within the callback itself
                    logger.error(f"Error in transcription_progress_wrapper: {cb_err}", exc_info=True)

            mp4_file.transcribe(
                model_name=whisper_config.get('model', 'large-v3'), # Use correct model from config
                language=whisper_config.get('language', 'auto'), # Use correct language setting
                initial_prompt=whisper_config.get('initial_prompt'),
                temperature=float(whisper_config.get('temperature', 0.0)),
                beam_size=int(whisper_config.get('beam_size', 5)),
                condition_on_previous=whisper_config.get('condition_on_previous', True),
                # device = 'cuda' if torch.cuda.is_available() else 'cpu', # Set device based on availability
                progress_callback=transcription_progress_wrapper
            )

            # 3. Run Scene Analysis (if enabled)
            if scene_analyzer_instance:
                 if progress_callback:
                      file_progress_callback(0.7, "Analyzing Scenes") # Mark start of scene analysis
                 
                 # Create directory for scene captures for this specific file
                 capture_dir_name = f"{os.path.splitext(file_basename)[0]}_captures"
                 capture_output_dir = os.path.join(output_folder, capture_dir_name)
                 # os.makedirs(capture_output_dir, exist_ok=True) # Directory creation moved to SceneAnalyzer

                 # Call run_scene_analysis without the progress_callback argument
                 mp4_file.run_scene_analysis(
                      frame_analysis_rate=int(scene_config.get('frame_analysis_rate', 30)),
                      capture_output_dir=capture_output_dir # Pass the capture directory
                 )
            else:
                 # If disabled, still mark this phase as complete for progress
                 if progress_callback:
                     file_progress_callback(0.9, "Skipped Scene Analysis")

            # Assign source filename and file index to each segment
            source_filename = os.path.basename(mp4_file_path)
            for segment in mp4_file.segments:
                segment.source_filename = source_filename
                segment.file_index = mp4_file.file_index # file_index is set during MP4File init
                segment.source_timecode_offset = mp4_file.timecode_offset 
            
            # Add processed segments from this file to the combined list
            all_segments.extend(mp4_file.segments)

            # 6. Save intermediate data for this file as JSON
            logger.info(f"Attempting to save intermediate JSON for {file_basename}...") # ADD LOG
            try:
                intermediate_data = mp4_file.get_intermediate_data_as_dict()
                logger.debug(f"Successfully generated intermediate data dict for {file_basename}.") # ADD LOG
                json_output_filename = f"{os.path.splitext(file_basename)[0]}_data.json"
                # Use the capture_output_dir if it exists, otherwise use the main output_folder
                json_base_dir = capture_output_dir if scene_analyzer_instance and os.path.exists(capture_output_dir) else output_folder # Check existence of capture_dir
                json_output_path = os.path.join(json_base_dir, json_output_filename)
                
                logger.debug(f"Determined JSON output path: {json_output_path}") # ADD LOG
                # Ensure the directory exists (especially if scene analysis was disabled but we still want json)
                os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
                
                with open(json_output_path, 'w', encoding='utf-8') as f_json:
                    json.dump(intermediate_data, f_json, ensure_ascii=False, indent=4)
                print(f"Intermediate data saved: {json_output_path}")
                logger.info(f"Successfully saved intermediate JSON: {json_output_path}") # ADD LOG
            except Exception as json_err:
                print(f"ERROR saving intermediate JSON for {file_basename}: {json_err}")
                logger.error(f"Failed to save intermediate JSON for {file_basename}", exc_info=True) # Enhanced logging

            files_processed_successfully += 1
            if progress_callback:
                 file_progress_callback(1.0, "Completed")

        except FileNotFoundError as e:
            print(f"ERROR processing {file_basename}: File not found - {e}")
            if progress_callback: file_progress_callback(1.0, "Error - File Not Found")
        except subprocess.CalledProcessError as e:
            print(f"ERROR processing {file_basename}: FFmpeg/FFprobe error - {e}")
            if progress_callback: file_progress_callback(1.0, "Error - FFmpeg/FFprobe")
        except Exception as e:
            print(f"ERROR processing {file_basename}: Unexpected error - {e}")
            # Add traceback for debugging
            import traceback
            traceback.print_exc()
            if progress_callback: file_progress_callback(1.0, f"Error - {type(e).__name__}")

    print(f"\n--- Processing Complete ({files_processed_successfully}/{total_files} successful) --- finalizing outputs ---")

    # Check if any segments were collected
    if not all_segments:
        print("No segments collected from any file. Skipping final EDL/SRT generation.")
        if progress_callback:
            progress_callback(1.0, "Finished - No Segments")
        return

    # Write combined EDL file using the new EDLData logic
    if progress_callback:
        progress_callback(0.95, "Generating Combined EDL")
    edl_output_path = os.path.join(output_folder, "combined_output.edl")
    try:
        # Initialize EDLData with converter and add all segments
        combined_edl_data = EDLData(title=edl_config.get('title', "Combined Project"), fcm=edl_config.get('fcm', "NON-DROP FRAME"), converter=converter)
        combined_edl_data.segments = all_segments # Assign collected segments
        
        with open(edl_output_path, "w", encoding="utf-8") as f:
            f.write(str(combined_edl_data)) # Uses the __str__ method which calls generate_edl_string
        print(f"Combined EDL file saved: {edl_output_path}")
    except Exception as e:
        print(f"ERROR saving combined EDL: {e}")
        traceback.print_exc()

    # Write combined SRT file using the new SRTData logic
    if progress_callback:
        progress_callback(0.98, "Generating Combined SRT")
    srt_output_path = os.path.join(output_folder, "combined_output.srt")
    try:
        # Initialize SRTData with converter and add all segments
        combined_srt_data = SRTData(converter=converter)
        combined_srt_data.segments = all_segments # Assign collected segments (already have offset info)
        
        combined_srt_data.write_to_file(srt_output_path) # This method now handles sorting and timestamp normalization
        print(f"Combined SRT file saved: {srt_output_path}")
    except Exception as e:
        print(f"ERROR saving combined SRT: {e}")
        traceback.print_exc()

    if progress_callback:
        progress_callback(1.0, "Finished")

def main():
    pass
    # Command-line argument parsing removed, assuming GUI or direct call to process_folder
    # parser = argparse.ArgumentParser(description="MP4 to EDL/SRT Converter")
    # ... (rest of argparse code) ...
    # args = parser.parse_args()
    # process_folder(args.input, args.output, use_timecode_offset=args.use_timecode)

if __name__ == "__main__":
    # This part is typically not run when used as a module by a GUI
    # If you want to run it standalone for testing:
    # print("Running main.py standalone (for testing)...")
    # test_input_folder = "./test_input" # Replace with your test folder
    # test_output_folder = "./test_output"
    # os.makedirs(test_input_folder, exist_ok=True)
    # os.makedirs(test_output_folder, exist_ok=True)
    # print(f"Please place test MP4 files in: {os.path.abspath(test_input_folder)}")
    # input("Press Enter to continue after placing files...")
    # process_folder(test_input_folder, test_output_folder)
    main()
