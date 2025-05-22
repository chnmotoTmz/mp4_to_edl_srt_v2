import os
import threading
import queue # Added for SSE
import json # Added for SSE
from flask import Flask, request, jsonify, Response # Added Response for SSE
from .main import process_folder, config_manager # Use actual imports

app = Flask(__name__)

# List to hold client message queues for SSE
client_queues = [] # Thread-safe operations using list.append and list.remove should be fine for this.
# A lock for modifying client_queues if more complex operations were needed.
# client_queues_lock = threading.Lock()


# --- Configuration Endpoint ---
@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify(config_manager.config)
    elif request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        config_manager.config.update(data)
        config_manager.save_config()
        return jsonify({"message": "Configuration updated successfully"})

# --- Processing Endpoint ---

def processing_progress_callback(stage, value, is_final_message=False):
    """
    Callback to send progress updates to all connected SSE clients.
    Expected progress_update structure: {"stage": "...", "value": ...}
    If is_final_message is True, a special message can be sent, or the value could be 100.
    """
    message = {"stage": stage, "value": value}
    if is_final_message:
        message["final"] = True # Indicate completion

    print(f"Progress Update: {message}") # Server-side log
    
    # Iterate over a copy of the list in case of modifications during iteration (though less likely here)
    # However, queue.put() is thread-safe, so direct iteration is also generally fine.
    # Using a lock would be more robust if adding/removing queues was very frequent and complex.
    for q in list(client_queues): # Iterate over a copy
        try:
            q.put(message)
        except Exception as e:
            # This might happen if a queue is somehow broken, though queue.Full is the typical blocking issue
            print(f"Error putting message to a client queue: {e}")
            # Potentially remove this queue if it's consistently problematic, but
            # queue removal is primarily handled by the /stream-progress endpoint.

@app.route('/process', methods=['POST'])
def handle_process():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    input_folder = data.get('input_folder')
    output_folder = data.get('output_folder')

    if not input_folder or not output_folder:
        return jsonify({"error": "input_folder and output_folder are required"}), 400
    
    if not os.path.isdir(input_folder):
        return jsonify({"error": f"Input folder not found: {input_folder}"}), 400
    
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except OSError as e:
        return jsonify({"error": f"Could not create output folder: {output_folder}, {str(e)}"}), 500

    processing_options = {
        "initial_prompt": data.get('initial_prompt', config_manager.config.get('initial_prompt')),
        "use_timecode_offset": data.get('use_timecode_offset', config_manager.config.get('use_timecode_offset')),
        "scene_analysis_enabled": data.get('scene_analysis_enabled', config_manager.config.get('scene_analysis_enabled')),
        "scene_analysis_rate": data.get('scene_analysis_rate', config_manager.config.get('scene_analysis_rate')),
    }
    processing_options = {k: v for k, v in processing_options.items() if v is not None}
    
    # Send an initial message to clients that processing has started
    processing_progress_callback("Processing", 0) 

    thread = threading.Thread(target=process_folder, args=(input_folder, output_folder, processing_progress_callback), kwargs=processing_options)
    thread.start()

    return jsonify({"message": "Processing started", "details": {"input": input_folder, "output": output_folder}})

# --- SSE Progress Stream Endpoint ---
@app.route('/stream-progress')
def stream_progress():
    def event_stream():
        client_q = queue.Queue()
        client_queues.append(client_q)
        print(f"Client connected. Total clients: {len(client_queues)}")
        try:
            while True:
                # Wait for a message from the queue. 
                # Using a timeout allows the server to check if the client is still connected
                # and to gracefully shut down the stream if the queue is empty for too long (e.g., processing finished).
                try:
                    message = client_q.get(timeout=30) # Timeout e.g. 30 seconds
                except queue.Empty:
                    # Timeout occurred, send a keep-alive comment or check connection
                    # A comment line in SSE starts with a colon
                    yield ": keep-alive\n\n"
                    continue 
                
                if message is None: # A way to signal stream closure from the callback, if needed.
                    break

                json_data = json.dumps(message)
                yield f"data: {json_data}\n\n"
                
                if message.get("final"): # If it's the last message, break the loop.
                    break
        except GeneratorExit:
            # This occurs if the client disconnects
            print("Client disconnected (GeneratorExit).")
        except Exception as e:
            print(f"Error in event stream: {e}")
        finally:
            if client_q in client_queues:
                client_queues.remove(client_q)
            print(f"Client queue removed. Total clients: {len(client_queues)}")

    return Response(event_stream(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
