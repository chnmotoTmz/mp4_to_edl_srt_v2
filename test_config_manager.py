import json
import os
import sys
import traceback

# Adjust path to import from the parent directory if mp4_to_edl_srt is there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

CONFIG_FILE_PATH = '/app/config.json' 

def run_test():
    print("TEST_SCRIPT: Starting ConfigManager direct test...")
    sys.stdout.flush()
    try:
        print("TEST_SCRIPT: Attempting to import ConfigManager...")
        sys.stdout.flush()
        from mp4_to_edl_srt.main import ConfigManager 
        print("TEST_SCRIPT: Successfully imported ConfigManager.")
        sys.stdout.flush()

        print(f"TEST_SCRIPT: Ensuring config file at: {CONFIG_FILE_PATH} has correct test structure.")
        sys.stdout.flush()
        
        # Define the config structure that matches ConfigManager.DEFAULT_CONFIG for relevant parts
        # This will be written to config.json to simulate a previously saved state
        test_config_content = {
            "whisper": { # Nested structure
                "initial_prompt": "Test Prompt from manual edit", # Test value
                "model": "base" # Overriding default 'large-v3'
            },
            "gui": { # Nested structure
                "ui_language": "en" # Test value
            },
            "paths": { # Example of other nested structure
                "last_input_folder": "/app/test_mp4_input",
                "last_output_folder": "/app/test_electron_output"
            },
            "scene_analysis": { # Example
                "enabled": True,
                "frame_analysis_rate": 5
            },
            "edl": { # Example
                 "use_timecode_offset": True
            },
            # Top-level keys that might be directly from Electron app's simpler config structure
            # These might get merged into respective sections if ConfigManager handles it, or stay top-level
            "input_folder": "/app/test_mp4_input", 
            "output_folder": "/app/test_electron_output",
            "initial_prompt": "Test Prompt from manual edit", # For flat access if needed by Electron part
            "use_timecode_offset": True, # For flat access
            "scene_analysis_enabled": True, # For flat access
            "scene_analysis_rate": 5, # For flat access
            "ui_language": "en", # For flat access
            "existing_config_value": "This should be preserved if save/load is partial" # Custom top-level
        }
        
        with open(CONFIG_FILE_PATH, 'w') as f:
            json.dump(test_config_content, f, indent=2)
        print(f"TEST_SCRIPT: Test config written to {CONFIG_FILE_PATH}.")
        sys.stdout.flush()

        config_manager_instance = ConfigManager(config_path=CONFIG_FILE_PATH) 
        print("TEST_SCRIPT: ConfigManager instantiated.")
        sys.stdout.flush()
        
        print(f"TEST_SCRIPT: Loaded config state after merge with defaults: {json.dumps(config_manager_instance.config, indent=2)}")
        sys.stdout.flush()

        # Test getting values (should come from the test_config_content via config.json)
        loaded_lang = config_manager_instance.get('gui', 'ui_language') 
        print(f"TEST_SCRIPT: Loaded gui.ui_language: {loaded_lang}")
        sys.stdout.flush()
        assert loaded_lang == "en", f"Expected ui_language 'en', got '{loaded_lang}'"

        loaded_prompt = config_manager_instance.get('whisper', 'initial_prompt')
        print(f"TEST_SCRIPT: Loaded whisper.initial_prompt: {loaded_prompt}")
        sys.stdout.flush()
        assert loaded_prompt == "Test Prompt from manual edit", f"Initial prompt mismatch: '{loaded_prompt}'"

        # Test setting values
        new_prompt_val = "Updated Prompt via Script"
        print(f"TEST_SCRIPT: Setting whisper.initial_prompt to: '{new_prompt_val}'")
        sys.stdout.flush()
        config_manager_instance.set('whisper', 'initial_prompt', new_prompt_val)
        
        new_model_val = "small" # This was 'whisper_model' top-level, now let's set it in 'whisper.model'
        print(f"TEST_SCRIPT: Setting whisper.model to: '{new_model_val}'")
        sys.stdout.flush()
        config_manager_instance.set('whisper', 'model', new_model_val) 

        print("TEST_SCRIPT: Saving configuration...")
        sys.stdout.flush()
        config_manager_instance.save_config()
        print("TEST_SCRIPT: Configuration saved.")
        sys.stdout.flush()

        # Verification by re-reading the file
        print(f"TEST_SCRIPT: Re-reading {CONFIG_FILE_PATH} for verification...")
        sys.stdout.flush()
        with open(CONFIG_FILE_PATH, 'r') as f:
            updated_config_data_from_file = json.load(f)
        
        print(f"TEST_SCRIPT: Data from file after save: {json.dumps(updated_config_data_from_file, indent=2)}")

        verified_prompt_from_file = updated_config_data_from_file.get('whisper', {}).get('initial_prompt')
        print(f"TEST_SCRIPT: Verified whisper.initial_prompt from file: {verified_prompt_from_file}")
        sys.stdout.flush()
        assert verified_prompt_from_file == new_prompt_val, f"Expected initial_prompt in file '{new_prompt_val}', got '{verified_prompt_from_file}'"

        verified_lang_from_file = updated_config_data_from_file.get('gui', {}).get('ui_language')
        print(f"TEST_SCRIPT: Verified gui.ui_language from file: {verified_lang_from_file}")
        sys.stdout.flush()
        assert verified_lang_from_file == "en", f"Expected ui_language in file 'en', got '{verified_lang_from_file}'"
        
        verified_model_from_file = updated_config_data_from_file.get('whisper', {}).get('model')
        print(f"TEST_SCRIPT: Verified whisper.model from file: {verified_model_from_file}")
        sys.stdout.flush()
        assert verified_model_from_file == new_model_val, f"Expected whisper.model in file '{new_model_val}', got '{verified_model_from_file}'"
        
        # Check if the custom top-level key was preserved (ConfigManager merges, doesn't fully overwrite with default structure)
        existing_val_from_file = updated_config_data_from_file.get('existing_config_value')
        print(f"TEST_SCRIPT: Verified existing_config_value from file: {existing_val_from_file}")
        sys.stdout.flush()
        assert existing_val_from_file == "This should be preserved if save/load is partial", "Custom top-level value was not preserved."
        
        # Check if flat keys used by Electron were preserved/merged (they might be if not clashing with DEFAULT_CONFIG sections)
        flat_ui_lang = updated_config_data_from_file.get('ui_language')
        print(f"TEST_SCRIPT: Verified top-level ui_language from file: {flat_ui_lang}")
        assert flat_ui_lang == "en", "Top-level ui_language was not preserved as expected."


        print("TEST_SCRIPT: ConfigManager direct test completed successfully!")
        sys.stdout.flush()

    except Exception as e:
        print(f"TEST_SCRIPT_ERROR: An error occurred during ConfigManager test: {e}")
        print(traceback.format_exc())
        sys.stdout.flush()
    finally:
        print("TEST_SCRIPT: End of ConfigManager direct test.")
        sys.stdout.flush()

if __name__ == '__main__':
    run_test()
