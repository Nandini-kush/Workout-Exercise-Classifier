# ============================================================
#  gui.py  –  Desktop GUI for Workout Exercise Classification
#  Run with:  python gui.py
#  Requires:  trained model at outputs/models/best_model.keras
# ============================================================

import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

from config import BEST_MODEL_PATH, REPORT_DIR, IMG_SIZE

# ── Colour palette ────────────────────────────────────────────
BG_DARK   = "#0d0d1a"
BG_CARD   = "#1a1a2e"
BG_INPUT  = "#12122a"
ACCENT    = "#4fc3f7"
ACCENT2   = "#f06292"
SUCCESS   = "#81c784"
WARNING   = "#ffb74d"
FG_MAIN   = "#e8e8ff"
FG_SUB    = "#9999cc"
BORDER    = "#2d2d50"
FONT_H1   = ("Courier New", 20, "bold")
FONT_H2   = ("Courier New", 13, "bold")
FONT_BODY = ("Courier New", 11)
FONT_SM   = ("Courier New", 9)
BTN_FONT  = ("Courier New", 11, "bold")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class WorkoutClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Workout Exercise Classifier")
        self.geometry("980x760")
        self.minsize(860, 620)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.model = None
        self.class_names = []

        self.current_img = None
        self.tk_img = None
        self.current_path = None
        self.is_video = False

        self.video_cap = None
        self.video_playing = False
        self.video_after_id = None

        self._load_model_and_classes()
        self._build_ui()

    # ── Model loading ─────────────────────────────────────────
    def _load_model_and_classes(self):
        class_path = os.path.join(REPORT_DIR, "class_names.json")

        if not os.path.exists(BEST_MODEL_PATH):
            messagebox.showerror(
                "Model Not Found",
                f"Trained model not found at:\n{BEST_MODEL_PATH}\n\nPlease run train.py first."
            )
            self.destroy()
            return

        try:
            self.model = tf.keras.models.load_model(BEST_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model:\n{e}")
            self.destroy()
            return

        if os.path.exists(class_path):
            with open(class_path, "r") as f:
                self.class_names = json.load(f)
        else:
            messagebox.showwarning(
                "Class Names Missing",
                f"class_names.json not found.\nExpected at:\n{class_path}\n\nPrediction labels may be wrong."
            )
            self.class_names = [f"Class_{i}" for i in range(self.model.output_shape[-1])]

    # ── UI construction ───────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG_DARK, pady=14)
        header.pack(fill="x")

        tk.Label(
            header,
            text="🏋 WORKOUT CLASSIFIER",
            font=FONT_H1,
            bg=BG_DARK,
            fg=ACCENT
        ).pack()

        tk.Label(
            header,
            text="AI-powered Exercise Recognition with Image / Video Preview",
            font=FONT_SM,
            bg=BG_DARK,
            fg=FG_SUB
        ).pack()

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20)

        # Main layout
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=20, pady=15)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Left card: preview
        left = self._card(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(left, text="Preview", font=FONT_H2, bg=BG_CARD, fg=ACCENT).pack(pady=(0, 8))

        self.img_label = tk.Label(
            left,
            bg=BG_INPUT,
            relief="flat",
            bd=0,
            text="No image or video selected\n\nSupports: JPG, PNG, BMP, GIF\nand MP4, AVI, MOV, MKV",
            fg=FG_SUB,
            font=FONT_BODY,
            width=40,
            height=18
        )
        self.img_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Video control row
        video_ctrl = tk.Frame(left, bg=BG_CARD)
        video_ctrl.pack(fill="x", pady=(10, 0))

        self.play_btn = self._btn(video_ctrl, "▶ Play", self._play_video, bg="#43a047", fg="white")
        self.play_btn.pack(side="left", padx=(0, 8))

        self.pause_btn = self._btn(video_ctrl, "⏸ Pause", self._pause_video, bg="#fb8c00", fg="white")
        self.pause_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = self._btn(video_ctrl, "⏹ Stop", self._stop_video, bg="#e53935", fg="white")
        self.stop_btn.pack(side="left")

        # Right side
        right = tk.Frame(main, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)

        # Controls
        ctrl_card = self._card(right)
        ctrl_card.pack(fill="x", pady=(0, 10))

        tk.Label(ctrl_card, text="Controls", font=FONT_H2, bg=BG_CARD, fg=ACCENT).pack(pady=(0, 10))

        self._btn(ctrl_card, "📂 Open Image / Video", self._open_file).pack(fill="x", pady=3)
        self._btn(ctrl_card, "🔍 Predict Exercise", self._run_predict, bg=ACCENT2).pack(fill="x", pady=3)
        self._btn(ctrl_card, "🗑 Clear", self._clear, bg="#333355", fg=FG_MAIN).pack(fill="x", pady=3)

        tk.Label(
            ctrl_card,
            text=f"Model loaded ✓\n{len(self.class_names)} classes",
            font=FONT_SM,
            bg=BG_CARD,
            fg=SUCCESS
        ).pack(pady=(10, 0))

        # Results
        res_card = self._card(right)
        res_card.pack(fill="both", expand=True)

        tk.Label(res_card, text="Prediction", font=FONT_H2, bg=BG_CARD, fg=ACCENT).pack(pady=(0, 10))

        self.pred_label = tk.Label(
            res_card,
            text="—",
            font=("Courier New", 17, "bold"),
            bg=BG_CARD,
            fg=FG_MAIN,
            wraplength=240
        )
        self.pred_label.pack()

        self.conf_label = tk.Label(
            res_card,
            text="Confidence: —",
            font=FONT_BODY,
            bg=BG_CARD,
            fg=FG_SUB
        )
        self.conf_label.pack(pady=(4, 12))

        tk.Label(res_card, text="Confidence", font=FONT_SM, bg=BG_CARD, fg=FG_SUB).pack(anchor="w", padx=8)

        self.conf_bar_bg = tk.Frame(res_card, bg="#222244", height=14)
        self.conf_bar_bg.pack(fill="x", padx=8, pady=(2, 12))

        self.conf_bar = tk.Frame(self.conf_bar_bg, bg=ACCENT, height=14, width=0)
        self.conf_bar.place(x=0, y=0, height=14)

        tk.Label(res_card, text="Top-3 Predictions", font=FONT_SM, bg=BG_CARD, fg=FG_SUB).pack(anchor="w", padx=8)

        self.top3_frame = tk.Frame(res_card, bg=BG_CARD)
        self.top3_frame.pack(fill="x", padx=8, pady=4)

        self.top3_labels = []
        for _ in range(3):
            lbl = tk.Label(
                self.top3_frame,
                text="",
                font=FONT_SM,
                bg=BG_CARD,
                fg=FG_SUB,
                anchor="w"
            )
            lbl.pack(fill="x", pady=1)
            self.top3_labels.append(lbl)

        # Status bar
        self.status_var = tk.StringVar(value="Ready • Open an image or video to begin")
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            font=FONT_SM,
            bg="#111128",
            fg=FG_SUB,
            anchor="w",
            padx=12,
            pady=6
        )
        status_bar.pack(fill="x", side="bottom")

    # ── Widget helpers ────────────────────────────────────────
    def _card(self, parent):
        return tk.Frame(
            parent,
            bg=BG_CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=14,
            pady=14
        )

    def _btn(self, parent, text, command, bg=ACCENT, fg=BG_DARK):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=BTN_FONT,
            bg=bg,
            fg=fg,
            relief="flat",
            cursor="hand2",
            activebackground=ACCENT2,
            activeforeground=BG_DARK,
            padx=10,
            pady=8
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT2, fg=BG_DARK))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg, fg=fg))
        return btn

    # ── File open / preview ───────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("Image/Video files", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self._stop_video(release_only=True)

        self.current_path = path
        ext = os.path.splitext(path)[1].lower()
        self.is_video = ext in VIDEO_EXTS

        self.status_var.set(f"Loading: {os.path.basename(path)}")
        self.update_idletasks()

        try:
            if self.is_video:
                self._show_first_video_frame(path)
            else:
                pil_img = Image.open(path).convert("RGB")
                self.current_img = pil_img
                self._show_preview(pil_img)

            self._clear_results()
            file_type = "Video" if self.is_video else "Image"
            self.status_var.set(f"{file_type} loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.status_var.set("Error loading file.")

    def _show_first_video_frame(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise IOError("Cannot read video frame.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        self.current_img = pil_img
        self._show_preview(pil_img)

    def _show_preview(self, pil_img):
        max_w, max_h = 420, 360
        w, h = pil_img.size
        ratio = min(max_w / w, max_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))

        display = pil_img.resize(new_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display)
        self.img_label.config(image=self.tk_img, text="", bg=BG_INPUT)

    def _display_cv_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        self._show_preview(pil_img)

    # ── Video playback ────────────────────────────────────────
    def _play_video(self):
        if not self.current_path or not self.is_video:
            messagebox.showinfo("No Video", "Please open a video first.")
            return

        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(self.current_path)

        self.video_playing = True
        self.status_var.set(f"Playing: {os.path.basename(self.current_path)}")
        self._update_video_frame()

    def _pause_video(self):
        if not self.is_video:
            return

        self.video_playing = False
        self.status_var.set("Video paused.")

    def _stop_video(self, release_only=False):
        self.video_playing = False

        if self.video_after_id is not None:
            try:
                self.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None

        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

        if not release_only and self.current_path and self.is_video:
            try:
                self._show_first_video_frame(self.current_path)
                self.status_var.set("Video stopped.")
            except Exception:
                self.status_var.set("Video stopped.")

    def _update_video_frame(self):
        if not self.video_playing or self.video_cap is None:
            return

        ret, frame = self.video_cap.read()

        if not ret or frame is None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()

        if ret and frame is not None:
            self._display_cv_frame(frame)
            self.video_after_id = self.after(30, self._update_video_frame)
        else:
            self._stop_video(release_only=True)
            self.status_var.set("Could not continue video playback.")

    # ── Prediction ────────────────────────────────────────────
    def _run_predict(self):
        if self.current_img is None:
            messagebox.showinfo("No Input", "Please open an image or video first.")
            return

        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded.")
            return

        self.status_var.set("Predicting...")
        self.update_idletasks()

        def _predict():
            try:
                img = self.current_img.resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)

                probs = self.model.predict(arr, verbose=0)[0]
                top3_idx = np.argsort(probs)[::-1][:3]

                top_idx = top3_idx[0]
                top_label = self.class_names[top_idx]
                top_conf = float(probs[top_idx])

                self.after(0, self._update_results, top_label, top_conf, probs, top3_idx)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
                self.after(0, lambda: self.status_var.set("Prediction failed."))

        threading.Thread(target=_predict, daemon=True).start()

    def _update_results(self, label, confidence, probs, top3_idx):
        self.pred_label.config(text=label.upper(), fg=ACCENT)
        self.conf_label.config(text=f"Confidence: {confidence * 100:.1f}%")

        self.conf_bar_bg.update_idletasks()
        total_width = max(self.conf_bar_bg.winfo_width(), 1)
        bar_width = int(total_width * confidence)
        self.conf_bar.place(x=0, y=0, height=14, width=max(bar_width, 2))

        if confidence > 0.7:
            bar_color = SUCCESS
        elif confidence > 0.4:
            bar_color = WARNING
        else:
            bar_color = ACCENT2

        self.conf_bar.config(bg=bar_color)

        for i, idx in enumerate(top3_idx):
            cls = self.class_names[idx]
            prob = probs[idx] * 100
            icon = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            self.top3_labels[i].config(text=f"  {icon}  {cls:<22}  {prob:5.1f}%")

        self.status_var.set(f"Prediction: {label} ({confidence * 100:.1f}% confidence)")

    # ── Clear ─────────────────────────────────────────────────
    def _clear_results(self):
        self.pred_label.config(text="—", fg=FG_MAIN)
        self.conf_label.config(text="Confidence: —")
        self.conf_bar.place(width=0)
        for lbl in self.top3_labels:
            lbl.config(text="")

    def _clear(self):
        self._stop_video(release_only=True)

        self.current_img = None
        self.tk_img = None
        self.current_path = None
        self.is_video = False

        self.img_label.config(
            image="",
            text="No image or video selected\n\nSupports: JPG, PNG, BMP, GIF\nand MP4, AVI, MOV, MKV",
            fg=FG_SUB,
            bg=BG_INPUT
        )

        self._clear_results()
        self.status_var.set("Ready • Open an image or video to begin")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    app = WorkoutClassifierApp()
    app.mainloop()