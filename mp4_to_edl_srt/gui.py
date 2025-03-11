import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
import sys
import glob
import json

from main import process_folder


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
        
        # 言語設定を読み込む（もし以前に保存されていれば）
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        self.load_settings()
        
        self.root.title(self.get_text("title"))
        self.root.geometry("800x900")
        self.root.resizable(True, True)
        
        self.input_folder = tk.StringVar(value=os.path.join(os.getcwd(), "input_mp4_files"))
        self.output_folder = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.initial_prompt = tk.StringVar(value=self.get_text("initial_prompt_default"))
        self.use_timecode_offset = tk.BooleanVar(value=True)
        
        self.create_widgets()
        
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
        
    def save_settings(self):
        """設定をJSONファイルに保存"""
        try:
            settings = {
                "language": self.language.get(),
                "last_input_folder": self.input_folder.get(),
                "last_output_folder": self.output_folder.get()
            }
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"設定の保存中にエラーが発生しました: {e}")
    
    def load_settings(self):
        """設定をJSONファイルから読み込み"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    if "language" in settings:
                        self.language.set(settings["language"])
                    if "last_input_folder" in settings and os.path.exists(settings["last_input_folder"]):
                        self.input_folder = tk.StringVar(value=settings["last_input_folder"])
                    if "last_output_folder" in settings and os.path.exists(settings["last_output_folder"]):
                        self.output_folder = tk.StringVar(value=settings["last_output_folder"])
        except Exception as e:
            print(f"設定の読み込み中にエラーが発生しました: {e}")
    
    def change_language(self, *args):
        """言語変更時の処理"""
        # 設定を保存
        self.save_settings()
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
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(options_frame, text=self.get_text("use_timecode"), variable=self.use_timecode_offset).pack(anchor=tk.W, padx=10, pady=5)
        
        # ログエリア
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text=self.get_text("cancel"), command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text=self.get_text("start_conversion"), command=self.start_conversion).pack(side=tk.RIGHT)

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title=self.get_text("input_folder"))
        if folder:
            self.input_folder.set(folder)
            self.save_settings()

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title=self.get_text("output_folder"))
        if folder:
            self.output_folder.set(folder)
            self.save_settings()

    def check_mp4_files(self, folder):
        if not os.path.exists(folder):
            return []
        return glob.glob(os.path.join(folder, "*.mp4")) + glob.glob(os.path.join(folder, "*.MP4"))

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_conversion(self):
        input_folder = self.input_folder.get()
        output_folder = self.output_folder.get()
        
        # 入力フォルダの存在確認
        if not os.path.exists(input_folder):
            messagebox.showerror(self.get_text("error"), self.get_text("input_folder_not_exist"))
            return
        
        # MP4ファイルの存在確認
        mp4_files = self.check_mp4_files(input_folder)
        if not mp4_files:
            messagebox.showerror(self.get_text("error"), self.get_text("no_mp4_files"))
            return
        
        # 確認ダイアログ
        if not messagebox.askyesno(self.get_text("confirm_title"), self.get_text("confirm_message")):
            return
        
        # 処理開始のログ
        self.log_text.delete(1.0, tk.END)
        self.log(f"{self.get_text('processing')}")
        
        # 環境変数にプロンプトを設定
        os.environ["WHISPER_INITIAL_PROMPT"] = self.initial_prompt.get()
        
        # 別スレッドで処理を実行
        threading.Thread(
            target=self.run_conversion,
            args=(input_folder, output_folder, self.use_timecode_offset.get()),
            daemon=True
        ).start()

    def run_conversion(self, input_folder, output_folder, use_timecode_offset=True):
        try:
            # 標準出力をリダイレクトするクラス
            class StdoutRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                    self.buffer = ""
                    self._old_stdout = sys.stdout
                
                def write(self, string):
                    self.buffer += string
                    if "\n" in self.buffer:
                        lines = self.buffer.split("\n")
                        for line in lines[:-1]:
                            self.text_widget.insert(tk.END, line + "\n")
                            self.text_widget.see(tk.END)
                            self.text_widget.update()
                        self.buffer = lines[-1]
                    
                    # 元の標準出力にも書き込む
                    self._old_stdout.write(string)
                
                def flush(self):
                    if self.buffer:
                        self.text_widget.insert(tk.END, self.buffer)
                        self.text_widget.see(tk.END)
                        self.text_widget.update()
                        self.buffer = ""
                    self._old_stdout.flush()
            
            # 標準出力をリダイレクト
            old_stdout = sys.stdout
            sys.stdout = StdoutRedirector(self.log_text)
            
            try:
                # 処理実行
                process_folder(input_folder, output_folder, use_timecode_offset)
                
                # 処理完了通知（GUIスレッドで実行）
                self.root.after(0, lambda: self.conversion_completed(output_folder))
            finally:
                # 標準出力を元に戻す
                sys.stdout = old_stdout
                
        except Exception as e:
            # エラー通知（GUIスレッドで実行）
            self.root.after(0, lambda: messagebox.showerror(self.get_text("error"), str(e)))

    def conversion_completed(self, output_folder):
        self.log(f"\n{self.get_text('completed')}")
        self.log(f"{self.get_text('output_files')} {output_folder}")
        messagebox.showinfo(self.get_text("completed"), f"{self.get_text('output_files')} {output_folder}")


def main():
    root = tk.Tk()
    app = MP4ToEDLSRTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 