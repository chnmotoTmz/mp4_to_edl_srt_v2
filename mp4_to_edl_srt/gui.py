import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import sys
import glob

from main import process_folder


class MP4ToEDLSRTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MP4 to EDL/SRT Converter")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.input_folder = tk.StringVar(value=os.path.join(os.getcwd(), "input_mp4_files"))
        self.output_folder = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.initial_prompt = tk.StringVar(value="日本語での自然な会話。文脈に応じて適切な表現を使用してください。")
        
        self.create_widgets()
        
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # タイトル
        title_label = ttk.Label(main_frame, text="MP4 to EDL/SRT Converter", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 入力フォルダ選択
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="入力フォルダ:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_folder, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="参照...", command=self.browse_input_folder).pack(side=tk.LEFT, padx=(10, 0))
        
        # 出力フォルダ選択
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="出力フォルダ:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_folder, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="参照...", command=self.browse_output_folder).pack(side=tk.LEFT, padx=(10, 0))
        
        # 初期プロンプト設定
        self.prompt_frame = ttk.Frame(main_frame)
        self.prompt_frame.pack(fill=tk.X, pady=(20, 5))
        
        ttk.Label(self.prompt_frame, text="初期プロンプト:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(self.prompt_frame, textvariable=self.initial_prompt, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 音声前処理オプション
        options_frame = ttk.LabelFrame(main_frame, text="音声前処理オプション")
        options_frame.pack(fill=tk.X, pady=(20, 5))
        
        # 音声前処理の有効/無効
        self.enable_preprocessing = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="音声前処理を有効にする（ノイズ除去と音量正規化）", 
                        variable=self.enable_preprocessing).pack(anchor=tk.W, padx=10, pady=5)
        
        # Whisperパラメータ設定
        params_frame = ttk.LabelFrame(main_frame, text="Whisperパラメータ")
        params_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Temperature設定
        temp_frame = ttk.Frame(params_frame)
        temp_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT, padx=(0, 10))
        self.temperature = tk.DoubleVar(value=0.2)
        temp_scale = ttk.Scale(temp_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                              variable=self.temperature, length=200)
        temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(temp_frame, textvariable=self.temperature).pack(side=tk.LEFT, padx=(10, 0))
        
        # Beam Size設定
        beam_frame = ttk.Frame(params_frame)
        beam_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(beam_frame, text="Beam Size:").pack(side=tk.LEFT, padx=(0, 10))
        self.beam_size = tk.IntVar(value=5)
        beam_values = [1, 3, 5, 8, 10]
        beam_combo = ttk.Combobox(beam_frame, textvariable=self.beam_size, values=beam_values, width=5)
        beam_combo.pack(side=tk.LEFT)
        ttk.Label(beam_frame, text="（大きいほど精度が上がりますが、処理時間が長くなります）").pack(side=tk.LEFT, padx=(10, 0))
        
        # 文脈考慮設定
        context_frame = ttk.Frame(params_frame)
        context_frame.pack(fill=tk.X, padx=10, pady=5)
        self.condition_on_previous = tk.BooleanVar(value=True)
        ttk.Checkbutton(context_frame, text="前後の文脈を考慮する (condition_on_previous_text)", 
                        variable=self.condition_on_previous).pack(anchor=tk.W)
        
        # 進行状況
        self.progress_var = tk.DoubleVar()
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(20, 5))
        
        ttk.Label(progress_frame, text="進行状況:").pack(side=tk.LEFT, padx=(0, 10))
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100, mode="indeterminate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ログ表示エリア
        log_frame = ttk.LabelFrame(main_frame, text="ログ")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 20))
        
        self.log_text = tk.Text(log_frame, height=10, width=70, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 実行ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="キャンセル", command=self.root.destroy).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="変換開始", command=self.start_conversion).pack(side=tk.RIGHT)
    
    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="入力フォルダを選択")
        if folder:
            self.input_folder.set(folder)
            self.check_mp4_files(folder)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="出力フォルダを選択")
        if folder:
            self.output_folder.set(folder)
    
    def check_mp4_files(self, folder):
        """指定されたフォルダ内のMP4ファイルを確認し、結果をログに表示します"""
        try:
            mp4_files = glob.glob(os.path.join(folder, "*.mp4"))
            if mp4_files:
                self.log(f"入力フォルダ内に{len(mp4_files)}個のMP4ファイルが見つかりました:")
                for file in mp4_files:
                    self.log(f" - {os.path.basename(file)}")
            else:
                self.log(f"警告: 入力フォルダ '{folder}' にMP4ファイルが見つかりません")
        except Exception as e:
            self.log(f"フォルダの確認中にエラーが発生しました: {str(e)}")
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_conversion(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        initial_prompt = self.initial_prompt.get()
        
        # パスの正規化
        input_folder = os.path.normpath(input_folder)
        output_folder = os.path.normpath(output_folder)
        
        if not os.path.exists(input_folder):
            messagebox.showerror("エラー", f"入力フォルダが存在しません: {input_folder}")
            return
        
        # MP4ファイルの存在確認
        mp4_files = glob.glob(os.path.join(input_folder, "*.mp4"))
        if not mp4_files:
            result = messagebox.askquestion("警告", 
                f"入力フォルダ '{input_folder}' にMP4ファイルが見つかりません。\n処理を続行しますか？")
            if result != 'yes':
                return
        
        # 出力フォルダの作成
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                self.log(f"出力フォルダを作成しました: {output_folder}")
            except Exception as e:
                messagebox.showerror("エラー", f"出力フォルダの作成に失敗しました: {str(e)}")
                return
        
        # 進行状況バーの開始
        self.progress_bar.start()
        
        # ログのクリア
        self.log_text.delete(1.0, tk.END)
        self.log(f"入力フォルダ: {input_folder}")
        self.log(f"出力フォルダ: {output_folder}")
        if mp4_files:
            self.log(f"MP4ファイル数: {len(mp4_files)}")
            for file in mp4_files:
                self.log(f" - {os.path.basename(file)}")
        else:
            self.log("警告: MP4ファイルが見つかりません")
        self.log("変換を開始します...")
        
        # 環境変数に初期プロンプトを設定
        os.environ["WHISPER_INITIAL_PROMPT"] = initial_prompt
        self.log(f"初期プロンプトを設定: {initial_prompt}")
        
        # 音声前処理の設定を環境変数に設定
        os.environ["ENABLE_AUDIO_PREPROCESSING"] = str(self.enable_preprocessing.get())
        self.log(f"音声前処理: {'有効' if self.enable_preprocessing.get() else '無効'}")
        
        # Whisperパラメータを環境変数に設定
        os.environ["WHISPER_TEMPERATURE"] = str(self.temperature.get())
        os.environ["WHISPER_BEAM_SIZE"] = str(self.beam_size.get())
        os.environ["WHISPER_CONDITION_ON_PREVIOUS"] = str(self.condition_on_previous.get())
        
        self.log(f"Whisperパラメータ設定:")
        self.log(f" - Temperature: {self.temperature.get()}")
        self.log(f" - Beam Size: {self.beam_size.get()}")
        self.log(f" - 文脈考慮: {'有効' if self.condition_on_previous.get() else '無効'}")
        
        # 別スレッドで処理を実行
        thread = threading.Thread(target=self.run_conversion, args=(input_folder, output_folder))
        thread.daemon = True
        thread.start()
    
    def run_conversion(self, input_folder, output_folder):
        try:
            # 標準出力と標準エラー出力をキャプチャするためのリダイレクト
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            class StdoutRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                    self.buffer = ""
                
                def write(self, string):
                    self.buffer += string
                    if "\n" in self.buffer:
                        lines = self.buffer.split("\n")
                        for line in lines[:-1]:
                            self.text_widget.after(0, lambda msg=line: self.text_widget.insert(tk.END, msg + "\n"))
                            self.text_widget.after(0, self.text_widget.see, tk.END)
                        self.buffer = lines[-1]
                
                def flush(self):
                    if self.buffer:
                        self.text_widget.after(0, lambda msg=self.buffer: self.text_widget.insert(tk.END, msg + "\n"))
                        self.text_widget.after(0, self.text_widget.see, tk.END)
                        self.buffer = ""
            
            sys.stdout = StdoutRedirector(self.log_text)
            sys.stderr = StdoutRedirector(self.log_text)
            
            # 変換処理の実行
            self.log(f"処理を開始します: {input_folder} -> {output_folder}")
            process_folder(input_folder, output_folder)
            
            # 標準出力と標準エラー出力を元に戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # 処理完了後の処理
            self.root.after(0, self.conversion_completed, output_folder)
            
        except Exception as e:
            # エラーが発生した場合
            error_message = f"エラーが発生しました: {str(e)}"
            self.root.after(0, lambda: self.log(error_message))
            self.root.after(0, lambda: messagebox.showerror("エラー", error_message))
            self.root.after(0, self.progress_bar.stop)
    
    def conversion_completed(self, output_folder):
        self.progress_bar.stop()
        self.log("変換が完了しました！")
        
        # 出力ファイルの確認
        edl_path = os.path.join(output_folder, "output.edl")
        srt_path = os.path.join(output_folder, "output.srt")
        
        if os.path.exists(edl_path) and os.path.exists(srt_path):
            message = f"EDLファイルとSRTファイルが生成されました。\n\nEDL: {edl_path}\nSRT: {srt_path}\n\n出力フォルダを開きますか？"
            if messagebox.askyesno("完了", message):
                # 出力フォルダを開く
                try:
                    if sys.platform == 'win32':
                        os.startfile(output_folder)
                    elif sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', output_folder])
                    else:  # Linux
                        subprocess.run(['xdg-open', output_folder])
                except Exception as e:
                    self.log(f"出力フォルダを開く際にエラーが発生しました: {str(e)}")
        else:
            messagebox.showwarning("警告", "出力ファイルが見つかりません。ログを確認してください。")


if __name__ == "__main__":
    root = tk.Tk()
    app = MP4ToEDLSRTApp(root)
    root.mainloop() 