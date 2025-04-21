import os
import json
import glob
import argparse
import logging
from typing import List

# Assuming these modules are in the same parent directory or installed
from .segment import Segment
from .edl_data import EDLData
from .srt_data import SRTData
from .timecode_utils import TimecodeConverter
# from .config_manager import ConfigManager # May need if frame rate is in config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reconstruct_segments_from_json(json_file_path: str) -> List[Segment]:
    """Loads segment data from a JSON file and reconstructs Segment objects."""
    segments = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Adjust based on the actual JSON structure provided by the user
        if 'final_segments' in data:
            segment_dicts = data['final_segments']
            if not isinstance(segment_dicts, list):
                logger.error(f"'final_segments' key exists but is not a list in {json_file_path}. Skipping file.")
                return []
            if not segment_dicts:
                logger.info(f"Found 'final_segments' key but the list is empty in {json_file_path}. No segments to load from this file.")
                # Return empty list, but it's not an error
            else:
                 logger.info(f"Found {len(segment_dicts)} segment dictionaries in {os.path.basename(json_file_path)} under 'final_segments' key.")
        else:
            # Fallback logic if 'final_segments' key is missing
            logger.warning(f"Could not find a 'final_segments' key in {json_file_path}. Checking other possible keys or structures...")
            if isinstance(data.get('segments'), list):
                 segment_dicts = data.get('segments')
                 logger.info(f"Found segments under the 'segments' key in {json_file_path}.")
            elif isinstance(data, list):
                 segment_dicts = data
                 logger.info(f"Loaded segments directly from top-level list in {json_file_path}")
            else:
                 logger.error(f"Could not find segment list in expected locations within {json_file_path}. Skipping file.")
                 return []
            
            # Re-check if fallback resulted in empty list
            if not segment_dicts:
                 logger.info(f"Fallback check resulted in an empty segment list for {json_file_path}.")

        # Process the segment_dicts (which might be empty)
        for seg_dict in segment_dicts:
            # Ensure all required keys for Segment init are present, provide defaults if necessary
            # This prevents errors if older JSONs lack newer fields.
            # Adjust defaults as needed.
            seg_dict.setdefault('scene_id', None)
            seg_dict.setdefault('scene_description', None)
            seg_dict.setdefault('transcription_good_reason', None)
            seg_dict.setdefault('transcription_bad_reason', None)
            seg_dict.setdefault('source_timecode_offset', None)
            seg_dict.setdefault('source_filename', None)
            seg_dict.setdefault('file_index', None)
            
            # Check for essential keys before creating the segment
            if 'start_timecode' not in seg_dict or 'end_timecode' not in seg_dict or 'transcription' not in seg_dict:
                logger.warning(f"Skipping segment due to missing essential keys in {json_file_path}: {seg_dict}")
                continue

            try:
                segments.append(Segment(**seg_dict))
            except TypeError as e:
                logger.error(f"Error creating Segment object from dict in {json_file_path}: {e} - Dict: {seg_dict}")
            except Exception as e:
                 logger.error(f"Unexpected error creating Segment from dict in {json_file_path}: {e} - Dict: {seg_dict}")

    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {json_file_path}")
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {json_file_path}: {e}", exc_info=True)
        
    return segments

def regenerate_outputs(json_dir: str, output_dir: str):
    """
    Loads segments from all *_data.json files in json_dir,
    then regenerates combined EDL and SRT files in output_dir.
    """
    logger.info(f"Starting regeneration process.")
    logger.info(f"Searching for *_data.json files in: {json_dir}")
    
    # Change glob to search recursively
    json_files = glob.glob(os.path.join(json_dir, "**", "*_data.json"), recursive=True)
    
    if not json_files:
        logger.warning(f"No *_data.json files found recursively in {json_dir}. Nothing to process.")
        return

    logger.info(f"Found {len(json_files)} JSON files.")

    all_segments: List[Segment] = []
    for json_file in sorted(json_files): # Sort for deterministic order
        logger.info(f"Loading segments from: {os.path.basename(json_file)}")
        segments_from_file = reconstruct_segments_from_json(json_file)
        if segments_from_file:
            all_segments.extend(segments_from_file)
        else:
             logger.warning(f"No segments loaded from {os.path.basename(json_file)}.")


    if not all_segments:
        logger.error("No segments were loaded from any JSON file. Cannot generate output.")
        return

    logger.info(f"Total segments loaded: {len(all_segments)}")

    # --- Configuration ---
    # TODO: Consider loading frame rate from config.json if needed
    # For now, using default frame rate from TimecodeConverter
    try:
        converter = TimecodeConverter() # Assumes default frame rate (59.94) is okay
        # Optionally load from config:
        # config_manager = ConfigManager()
        # config = config_manager.config
        # frame_rate = config.get('output', {}).get('frame_rate', 59.94) # Example path
        # converter = TimecodeConverter(frame_rate=frame_rate)
        # logger.info(f"Using frame rate: {converter.frame_rate}")
    except Exception as e:
        logger.error(f"Failed to initialize TimecodeConverter: {e}", exc_info=True)
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Regenerate EDL ---
    edl_output_path = os.path.join(output_dir, "combined_output.edl")
    logger.info(f"Generating EDL file: {edl_output_path}")
    try:
        # Initialize EDLData with converter and add all segments
        combined_edl_data = EDLData(title="Regenerated Project", converter=converter)
        combined_edl_data.segments = all_segments # Assign collected segments
        
        with open(edl_output_path, "w", encoding="utf-8") as f:
            # Use the generate_edl_string method (or __str__) which now excludes comments
            f.write(str(combined_edl_data)) 
        logger.info(f"EDL file regenerated successfully: {edl_output_path}")
    except Exception as e:
        logger.error(f"ERROR generating combined EDL: {e}", exc_info=True)

    # --- Regenerate SRT ---
    srt_output_path = os.path.join(output_dir, "combined_output.srt")
    logger.info(f"Generating SRT file: {srt_output_path}")
    try:
        # Initialize SRTData with converter and add all segments
        combined_srt_data = SRTData(converter=converter)
        combined_srt_data.segments = all_segments # Assign collected segments
        
        # Use the write_to_file method which now excludes scene descriptions
        combined_srt_data.write_to_file(srt_output_path)
        # write_to_file prints its own success message
        logger.info(f"SRT file regeneration process completed for: {srt_output_path}")
    except Exception as e:
        logger.error(f"ERROR generating combined SRT: {e}", exc_info=True)

    logger.info("Regeneration finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate combined EDL and SRT files from existing intermediate JSON data.")
    parser.add_argument("json_dir", help="Directory containing the *_data.json files.")
    parser.add_argument("-o", "--output_dir", help="Directory to save the regenerated EDL and SRT files. Defaults to json_dir.")
    
    args = parser.parse_args()
    
    output_directory = args.output_dir if args.output_dir else args.json_dir
    
    regenerate_outputs(args.json_dir, output_directory) 