import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import sys
import glob
import json
import queue # Import queue for thread-safe communication

from mp4_to_edl_srt.main import process_folder, config_manager, converter # Import converter too if needed later


# 多言語対応のための翻訳データ
TRANSLATIONS = {
    "en": {  # 英語
        "title": "MP4 to EDL/SRT Converter",
        "input_folder": "Input Folder:",
        "output_folder": "Output Folder:",
        "browse": "Browse...",
        "initial_prompt": "Initial Prompt:",
        "initial_prompt_default": "Natural conversation in Japanese. Please use appropriate expressions according to the context.",
        "use_timecode": "Use internal timecode from MP4 files",
        "advanced_options": "Advanced Options",
        "start_conversion": "Start Conversion",
        "cancel": "Cancel",
        "language": "Language:",
        "processing": "Processing...",
        "completed": "Conversion completed!",
        "output_files": "Output files are saved to:",
        "error": "Error",
        "no_mp4_files": "No MP4 files found in the input folder",
        "input_folder_not_exist": "Input folder does not exist",
        "confirm_title": "Confirm",
        "confirm_message": "Are you sure you want to start the conversion?",
        "yes": "Yes",
        "no": "No",
        "processing_file": "Processing file"
    },
    "ja": {  # 日本語
        "title": "MP4 to EDL/SRT コンバーター",
        "input_folder": "入力フォルダ:",
        "output_folder": "出力フォルダ:",
        "browse": "参照...",
        "initial_prompt": "初期プロンプト:",
        "initial_prompt_default": "日本語での自然な会話。文脈に応じて適切な表現を使用してください。",
        "use_timecode": "MP4ファイルの内部タイムコードを使用",
        "advanced_options": "詳細オプション",
        "start_conversion": "変換開始",
        "cancel": "キャンセル",
        "language": "言語:",
        "processing": "処理中...",
        "completed": "変換完了！",
        "output_files": "出力ファイルの保存先:",
        "error": "エラー",
        "no_mp4_files": "入力フォルダにMP4ファイルが見つかりません",
        "input_folder_not_exist": "入力フォルダが存在しません",
        "confirm_title": "確認",
        "confirm_message": "変換を開始しますか？",
        "yes": "はい",
        "no": "いいえ",
        "processing_file": "処理中のファイル"
    }
}


class MP4ToEDLSRTApp:
    def __init__(self, root):
        self.root = root
        self.language = tk.StringVar(value="ja")  # デフォルト言語は日本語
        self.config = config_manager.config # Load initial config

        # Settings file handling can be removed or merged with config.json later
        # self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        # self.load_settings()

        self.root.title(self.get_text("title"))
        self.root.geometry("800x900") # Adjust size if needed
        self.root.resizable(True, True)

        # --- Tkinter Variables --- 
        self.input_folder = tk.StringVar(value=self.config.get('paths', {}).get('last_input_folder', os.path.join(os.getcwd(), "input_mp4_files")))
        self.output_folder = tk.StringVar(value=self.config.get('paths', {}).get('last_output_folder', os.path.join(os.getcwd(), "output")))
        # Load whisper prompt from config
        self.initial_prompt = tk.StringVar(value=self.config.get('whisper', {}).get('initial_prompt', self.get_text("initial_prompt_default")))
        # Load EDL offset from config
        self.use_timecode_offset = tk.BooleanVar(value=self.config.get('edl', {}).get('use_timecode_offset', True))
        # Load Scene Analysis settings from config
        self.scene_analysis_enabled = tk.BooleanVar(value=self.config.get('scene_analysis', {}).get('enabled', False))
        self.scene_analysis_rate = tk.IntVar(value=self.config.get('scene_analysis', {}).get('frame_analysis_rate', 30))

        self.processing_active = False # Flag to indicate if conversion is running
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")
        self.queue = queue.Queue() # Queue for thread-safe GUI updates

        self.create_widgets()
        self.check_queue() # Start checking the queue for updates

    def get_text(self, key):
        """現在の言語に基づいてテキストを取得"""
        lang = self.language.get()
        if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
            return TRANSLATIONS[lang][key]
        # フォールバック: 英語または最初に見つかったキー
        if "en" in TRANSLATIONS and key in TRANSLATIONS["en"]:
            return TRANSLATIONS["en"][key]
        for lang in TRANSLATIONS:
            if key in TRANSLATIONS[lang]:
                return TRANSLATIONS[lang][key]
        return key  # キーが見つからない場合はキー自体を返す
        
    def save_config(self):
        """Saves current GUI settings to config_manager and file"""
        self.config['paths']['last_input_folder'] = self.input_folder.get()
        self.config['paths']['last_output_folder'] = self.output_folder.get()
        self.config['whisper']['initial_prompt'] = self.initial_prompt.get()
        self.config['edl']['use_timecode_offset'] = self.use_timecode_offset.get()
        self.config['scene_analysis']['enabled'] = self.scene_analysis_enabled.get()
        try:
            # Validate analysis rate before saving
            rate = int(self.scene_analysis_rate.get())
            if rate <= 0:
                rate = 1 # Ensure positive rate
                self.scene_analysis_rate.set(rate)
            self.config['scene_analysis']['frame_analysis_rate'] = rate
        except ValueError:
             # Handle invalid input if needed, maybe reset to default
             default_rate = config_manager.DEFAULT_CONFIG['scene_analysis']['frame_analysis_rate']
             self.scene_analysis_rate.set(default_rate)
             self.config['scene_analysis']['frame_analysis_rate'] = default_rate

        config_manager.save_config() # Use ConfigManager to save
        print("Settings saved to config.json")

    def change_language(self, *args):
        """言語変更時の処理"""
        # 設定を保存
        self.save_config()
        # GUIを再構築
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_widgets()
    
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 言語選択（右上に配置）
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(0, 10), anchor=tk.NE)
        
        ttk.Label(lang_frame, text=self.get_text("language")).pack(side=tk.LEFT, padx=(0, 5))
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.language, state="readonly", width=10)
        lang_combo["values"] = ["en", "ja"]
        lang_combo.pack(side=tk.LEFT)
        # 言語変更時のイベント
        self.language.trace_add("write", self.change_language)
        
        # タイトル
        title_label = ttk.Label(main_frame, text=self.get_text("title"), font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 入力フォルダ選択
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text=self.get_text("input_folder")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_folder, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text=self.get_text("browse"), command=self.browse_input_folder).pack(side=tk.LEFT, padx=(10, 0))
        
        # 出力フォルダ選択
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text=self.get_text("output_folder")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_folder, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text=self.get_text("browse"), command=self.browse_output_folder).pack(side=tk.LEFT, padx=(10, 0))
        
        # 初期プロンプト設定
        prompt_frame = ttk.Frame(main_frame)
        prompt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(prompt_frame, text=self.get_text("initial_prompt")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(prompt_frame, textvariable=self.initial_prompt, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 詳細オプション
        options_frame = ttk.LabelFrame(main_frame, text=self.get_text("advanced_options"))
        options_frame.pack(fill=tk.X, pady=10, padx=5)

        # EDL Timecode Offset Checkbox
        edl_options_frame = ttk.Frame(options_frame)
        edl_options_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Checkbutton(edl_options_frame, text=self.get_text("use_timecode"), variable=self.use_timecode_offset).pack(side=tk.LEFT)

        # Scene Analysis Options Frame
        scene_options_frame = ttk.Frame(options_frame)
        scene_options_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Checkbutton(scene_options_frame, text=self.get_text("enable_scene_analysis"), variable=self.scene_analysis_enabled, command=self.toggle_scene_rate_entry).pack(side=tk.LEFT)

        self.scene_rate_label = ttk.Label(scene_options_frame, text=self.get_text("scene_analysis_rate"))
        self.scene_rate_label.pack(side=tk.LEFT, padx=(10, 2))

        self.scene_rate_entry = ttk.Spinbox(scene_options_frame, from_=1, to=999, increment=1, textvariable=self.scene_analysis_rate, width=5)
        self.scene_rate_entry.pack(side=tk.LEFT)
        self.toggle_scene_rate_entry() # Set initial state

        # ログエリア
        log_frame = ttk.LabelFrame(main_frame, text=self.get_text("log_output")) # Changed label
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # --- Progress Bar and Status Label --- 
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(5,0), padx=5)

        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(fill=tk.X, expand=True, side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=1.0)
        self.progress_bar.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(10, 0))

        # --- Buttons --- 
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=5)

        self.cancel_button = ttk.Button(button_frame, text=self.get_text("cancel"), command=self.cancel_conversion, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        self.start_button = ttk.Button(button_frame, text=self.get_text("start_conversion"), command=self.start_conversion)
        self.start_button.pack(side=tk.RIGHT)

    def toggle_scene_rate_entry(self):
        """Enable/disable scene analysis rate entry based on checkbox"""
        if self.scene_analysis_enabled.get():
            self.scene_rate_label.config(state=tk.NORMAL)
            self.scene_rate_entry.config(state=tk.NORMAL)
        else:
            self.scene_rate_label.config(state=tk.DISABLED)
            self.scene_rate_entry.config(state=tk.DISABLED)

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title=self.get_text("input_folder"))
        if folder:
            self.input_folder.set(folder)
            self.save_config()

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title=self.get_text("output_folder"))
        if folder:
            self.output_folder.set(folder)
            self.save_config()

    def check_mp4_files(self, folder):
        if not os.path.exists(folder):
            return []
        return glob.glob(os.path.join(folder, "*.mp4")) + glob.glob(os.path.join(folder, "*.MP4"))

    def log(self, message):
        # This is now handled by the StdoutRedirector or progress callback
        # We might still want a way to add specific GUI messages
        self.update_gui(message + "\n", None)
        pass

    def update_gui(self, status_text, progress_value):
        """Thread-safe method to update status label and progress bar"""
        if status_text is not None:
             self.status_var.set(status_text)
             # Add to log text as well
             self.log_text.insert(tk.END, status_text + "\n")
             self.log_text.see(tk.END)
        if progress_value is not None:
             self.progress_var.set(progress_value)
        self.root.update_idletasks()

    def check_queue(self):
         """ Check the queue for messages from the worker thread. """
         try:
             message = self.queue.get_nowait() # Check for message without blocking
             status, progress = message
             self.update_gui(status, progress)
         except queue.Empty:
             pass # No message in the queue
         finally:
             # Schedule the next check
             self.root.after(100, self.check_queue)

    def gui_progress_callback(self, value: float, stage: str):
        """Callback function passed to process_folder, puts updates in the queue."""
        # Put updates into the queue for the main thread to process
        self.queue.put((stage, value))

    def start_conversion(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()

        if self.processing_active:
            messagebox.showwarning("Warning", "Processing is already in progress.")
            return

        # 入力フォルダの存在確認
        if not os.path.exists(input_folder):
            messagebox.showerror(self.get_text("error"), self.get_text("input_folder_not_exist"))
            return
        
        # MP4ファイルの存在確認
        mp4_files = self.check_mp4_files(input_folder)
        if not mp4_files:
            messagebox.showerror(self.get_text("error"), self.get_text("no_mp4_files"))
            return
        
        # Save current settings to config file before starting
        self.save_config()

        # Confirmation dialog
        if not messagebox.askyesno(self.get_text("confirm_title"), self.get_text("confirm_message")):
            return

        # Clear log and reset progress
        self.log_text.delete(1.0, tk.END)
        self.update_gui("Starting...", 0.0)
        self.processing_active = True
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)

        # Run conversion in a separate thread
        self.conversion_thread = threading.Thread(
            target=self.run_conversion,
            args=(input_folder, output_folder),
            daemon=True
        )
        self.conversion_thread.start()

    def run_conversion(self, input_folder, output_folder):
        # Note: use_timecode_offset is read from config inside process_folder
        try:
            # No longer need StdoutRedirector if progress callback is comprehensive
            # Process folder and pass the GUI callback method
            process_folder(input_folder, output_folder, progress_callback=self.gui_progress_callback)

            # Signal completion (put final message in queue)
            self.queue.put(("Finished", 1.0))
            # Optional: Add final message directly for immediate display after thread finishes
            # self.root.after(0, lambda: self.conversion_completed(output_folder))

        except Exception as e:
            # Signal error (put error message in queue)
            error_message = f"ERROR: {e}"
            import traceback
            error_trace = traceback.format_exc()
            print(error_trace) # Print full traceback to console for debugging
            self.queue.put((error_message, None)) # Update status with error
            # self.root.after(0, lambda msg=error_message: messagebox.showerror(self.get_text("error"), msg))
        finally:
            # Signal that processing is no longer active
            self.queue.put(("Idle", None)) # Custom signal or use status text
            self.processing_active = False
            # Re-enable start button, disable cancel via queue or root.after
            self.queue.put(("EnableStart", None)) # Use a special message or handle in check_queue

    def cancel_conversion(self):
        # Cancellation is tricky with external processes like ffmpeg/whisper
        # A simple approach is to set a flag and let the thread finish its current step
        # A more complex approach involves terminating the thread/process, which is risky
        if self.processing_active:
             print("Cancellation requested (Note: May not stop immediately)")
             # Set a flag that the processing loop can check (needs implementation in process_folder/MP4File)
             # Or simply inform the user cancellation is requested but not guaranteed
             self.status_var.set("Cancellation Requested...")
             # For now, just re-enable buttons - process will finish in background
             self.start_button.config(state=tk.NORMAL)
             self.cancel_button.config(state=tk.DISABLED)
             self.processing_active = False # Allow starting again
        else:
             self.cancel_button.config(state=tk.DISABLED)

    def conversion_completed(self, output_folder):
        # This is now primarily handled by the progress updates and final status message
        self.update_gui(f"{self.get_text('completed')} Output: {output_folder}", 1.0)
        messagebox.showinfo(self.get_text("completed"), f"{self.get_text('output_files')} {output_folder}")
        # Reset buttons here if not handled via queue
        # self.start_button.config(state=tk.NORMAL)
        # self.cancel_button.config(state=tk.DISABLED)
        # self.processing_active = False

def main():
    root = tk.Tk()
    app = MP4ToEDLSRTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 