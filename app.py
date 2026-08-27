"""Windows desktop app that turns video/audio into a cleaned-up SRT subtitle file."""

import os
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from subtitle_core import (
    DEFAULT_MODEL,
    LANGUAGES,
    MODEL_CHOICES,
    MODEL_SIZES,
    TERMS_PATH,
    TermReplacer,
    build_srt,
    configure_utf8_console,
    format_clock,
    load_terms,
    postprocess_segments,
    probe_duration,
    truncate_middle,
    whisper_cache_dir,
    whisper_progress,
)

configure_utf8_console()

COLOR_GPU = "#2FA572"
COLOR_CPU = "#D97706"
COLOR_MUTED = "#9AA0A6"
COLOR_TEXT = "#DCE4EE"


# ======================================================================================
# GUI
# ======================================================================================

class SubtitleApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Whisper Subtitle Generator")
        self.geometry("560x420")
        self.resizable(False, False)

        self.video_path = None
        self.output_dir = None
        self.output_override = False
        self.last_output_dir = None

        self._backend_lock = threading.Lock()
        self._torch = None
        self._whisper = None
        self._device = "cpu"
        self._device_label = None
        self._model = None
        self._model_key = None
        self._running = False
        self._indeterminate = False

        self._build_ui()
        self._ensure_terms_file()

        # torch takes seconds to import, so probe the GPU off the main thread and let
        # the window paint immediately.
        threading.Thread(target=self._detect_device, daemon=True).start()

    # -- layout ---------------------------------------------------------------------

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=18, pady=14)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        header = ctk.CTkFrame(container, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Whisper Subtitle Generator",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self.badge = ctk.CTkLabel(
            header,
            text="Detecting...",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3A3A3A",
            corner_radius=10,
            padx=10,
            pady=3,
        )
        self.badge.grid(row=0, column=1, sticky="e")

        self.select_button = ctk.CTkButton(
            container, text="Select Video", height=34, command=self._on_select_video
        )
        self.select_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 4))

        self.video_label = ctk.CTkLabel(
            container, text="No file selected", text_color=COLOR_MUTED, anchor="w"
        )
        self.video_label.grid(row=2, column=0, columnspan=2, sticky="ew")

        ctk.CTkLabel(container, text="Language", anchor="w").grid(
            row=3, column=0, sticky="w", pady=(10, 2)
        )
        ctk.CTkLabel(container, text="Model", anchor="w").grid(
            row=3, column=1, sticky="w", padx=(10, 0), pady=(10, 2)
        )

        self.language_selector = ctk.CTkSegmentedButton(
            container, values=list(LANGUAGES), height=32
        )
        self.language_selector.set("Turkish")
        self.language_selector.grid(row=4, column=0, sticky="ew")

        self.model_selector = ctk.CTkComboBox(
            container, values=MODEL_CHOICES, state="readonly", height=32
        )
        self.model_selector.set(DEFAULT_MODEL)
        self.model_selector.grid(row=4, column=1, sticky="ew", padx=(10, 0))

        self.output_button = ctk.CTkButton(
            container, text="Output Folder", height=32, command=self._on_select_output
        )
        self.output_button.grid(row=5, column=0, sticky="ew", pady=(12, 4))

        self.terms_button = ctk.CTkButton(
            container,
            text="Edit Terms",
            height=32,
            fg_color="#3A3A3A",
            hover_color="#4A4A4A",
            command=self._on_edit_terms,
        )
        self.terms_button.grid(row=5, column=1, sticky="ew", padx=(10, 0), pady=(12, 4))

        self.output_label = ctk.CTkLabel(
            container,
            text="Output: same folder as the video",
            text_color=COLOR_MUTED,
            anchor="w",
        )
        self.output_label.grid(row=6, column=0, columnspan=2, sticky="ew")

        self.generate_button = ctk.CTkButton(
            container,
            text="Generate Subtitles",
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
            command=self._on_generate,
        )
        self.generate_button.grid(row=7, column=0, sticky="ew", pady=(14, 8))

        self.open_button = ctk.CTkButton(
            container,
            text="Open Folder",
            height=38,
            fg_color="#3A3A3A",
            hover_color="#4A4A4A",
            state="disabled",
            command=self._on_open_folder,
        )
        self.open_button.grid(row=7, column=1, sticky="ew", padx=(10, 0), pady=(14, 8))

        self.progress = ctk.CTkProgressBar(container, height=12, mode="determinate")
        self.progress.set(0)
        self.progress.grid(row=8, column=0, columnspan=2, sticky="ew")

        self.status_label = ctk.CTkLabel(
            container, text="Select a video to begin.", text_color=COLOR_MUTED, anchor="w"
        )
        self.status_label.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self._inputs = [
            self.select_button,
            self.language_selector,
            self.model_selector,
            self.output_button,
            self.terms_button,
            self.generate_button,
        ]

    # -- thread-safe UI updates -------------------------------------------------------

    def _post(self, func, *args, **kwargs):
        """Run a UI update on the main thread - Tk widgets are not thread-safe."""
        self.after(0, lambda: func(*args, **kwargs))

    def _set_status(self, text, color=COLOR_MUTED):
        self.status_label.configure(text=text, text_color=color)

    # -- device detection --------------------------------------------------------------

    def _ensure_backend(self):
        """Import torch/whisper once and resolve the device. Thread-safe."""
        with self._backend_lock:
            if self._whisper is None:
                import torch
                import whisper

                self._torch = torch
                self._whisper = whisper
                if torch.cuda.is_available():
                    self._device = "cuda"
                    name = torch.cuda.get_device_name(0)
                    self._device_label = name.replace("NVIDIA GeForce ", "").strip()
                else:
                    self._device = "cpu"
                    self._device_label = None
            return self._device

    def _detect_device(self):
        try:
            self._ensure_backend()
        except Exception as exc:  # torch missing or CUDA driver failure
            self._post(self.badge.configure, text="Backend error", fg_color=COLOR_CPU)
            self._post(self._set_status, f"Could not load PyTorch: {exc}", COLOR_CPU)
            return

        if self._device == "cuda":
            self._post(
                self.badge.configure,
                text=f"CUDA · {self._device_label}",
                fg_color=COLOR_GPU,
            )
        else:
            self._post(self.badge.configure, text="CPU · slow", fg_color=COLOR_CPU)

    # -- actions ------------------------------------------------------------------------

    def _on_select_video(self):
        path = filedialog.askopenfilename(
            title="Select a video or audio file",
            filetypes=[
                ("Video/Audio", "*.mp4 *.mov *.mkv *.avi *.mp3 *.wav"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.video_path = Path(path)
        self.video_label.configure(
            text=truncate_middle(self.video_path.name), text_color=COLOR_TEXT
        )
        if not self.output_override:
            self.output_dir = self.video_path.parent
            self.output_label.configure(
                text="Output: " + truncate_middle(str(self.output_dir))
            )
        self.generate_button.configure(state="normal")
        self._set_status("Ready.")

    def _on_select_output(self):
        initial = str(self.output_dir) if self.output_dir else None
        path = filedialog.askdirectory(title="Select output folder", initialdir=initial)
        if not path:
            return
        self.output_dir = Path(path)
        self.output_override = True
        self.output_label.configure(text="Output: " + truncate_middle(str(self.output_dir)))

    def _ensure_terms_file(self):
        """Create terms.json up front so Edit Terms always has something to open."""
        if not TERMS_PATH.exists():
            try:
                TERMS_PATH.write_text("{}\n", encoding="utf-8")
            except OSError:
                pass

    def _on_edit_terms(self):
        self._ensure_terms_file()
        try:
            os.startfile(str(TERMS_PATH))
        except OSError as exc:
            messagebox.showerror("Edit Terms", f"Could not open terms.json:\n{exc}")

    def _on_open_folder(self):
        target = self.last_output_dir or self.output_dir
        if not target:
            return
        try:
            os.startfile(str(target))
        except OSError as exc:
            messagebox.showerror("Open Folder", f"Could not open the folder:\n{exc}")

    def _on_generate(self):
        if self._running or not self.video_path:
            return
        if not self.video_path.exists():
            messagebox.showerror("Generate Subtitles", "The selected file no longer exists.")
            return

        output_dir = self.output_dir or self.video_path.parent
        language = self.language_selector.get() or "Turkish"
        lang_code = LANGUAGES.get(language, "tr")
        model_name = self.model_selector.get() or DEFAULT_MODEL

        self._set_running(True)
        self.progress.set(0)
        self.open_button.configure(state="disabled")
        self._set_status("Loading model...")

        threading.Thread(
            target=self._worker,
            args=(self.video_path, output_dir, lang_code, model_name),
            daemon=True,
        ).start()

    # -- worker ---------------------------------------------------------------------------

    def _worker(self, video, output_dir, lang_code, model_name):
        try:
            self._ensure_backend()
            duration = probe_duration(video)

            if not self._is_model_cached(model_name):
                self._post(self._start_indeterminate, model_name)

            model = self._load_model(model_name)
            self._post(self._stop_indeterminate)
            self._post(self._set_status, "Transcribing... 0%")

            def on_progress(fraction):
                self._post(self._update_progress, fraction, duration)

            with whisper_progress(on_progress):
                result = model.transcribe(
                    str(video),
                    language=lang_code,
                    fp16=(self._device == "cuda"),
                    verbose=False,  # inverted flag: False enables Whisper's tqdm bar
                )

            self._post(self._set_status, "Writing SRT...")

            terms, warning = load_terms()
            if warning:
                self._post(messagebox.showwarning, "terms.json", warning)

            segments = postprocess_segments(
                result.get("segments", []), TermReplacer(terms, lang_code)
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video.stem}_{lang_code}.srt"
            output_path.write_text(build_srt(segments), encoding="utf-8")

            self._post(self._on_success, output_path, len(segments))
        except Exception as exc:
            self._post(self._on_error, exc)
        finally:
            self._post(self._set_running, False)

    def _is_model_cached(self, model_name):
        """True when the weights are already on disk, so no download progress is needed."""
        try:
            url = self._whisper._MODELS[model_name]
            return os.path.exists(os.path.join(whisper_cache_dir(), os.path.basename(url)))
        except Exception:
            return True  # unknown: assume cached and keep the determinate bar

    def _load_model(self, model_name):
        """Single-slot cache: reuse the same model, free VRAM when switching."""
        key = (model_name, self._device)
        if self._model is not None and self._model_key == key:
            return self._model

        self._model = None
        self._model_key = None
        if self._device == "cuda":
            self._torch.cuda.empty_cache()

        model = self._whisper.load_model(model_name, device=self._device)
        self._model = model
        self._model_key = key
        return model

    # -- UI state -------------------------------------------------------------------------

    def _set_running(self, running):
        self._running = running
        state = "disabled" if running else "normal"
        for widget in self._inputs:
            widget.configure(state=state)
        if not running:
            # Restore the combo box to readonly - a plain "normal" state would let the
            # user type an arbitrary model name into it.
            self.model_selector.configure(state="readonly")
            self._stop_indeterminate()
            if not self.video_path:
                self.generate_button.configure(state="disabled")

    def _start_indeterminate(self, model_name):
        size = MODEL_SIZES.get(model_name, "")
        self._indeterminate = True
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self._set_status(f"Downloading {model_name} ({size}, first run only)...")

    def _stop_indeterminate(self):
        if not self._indeterminate:
            return
        self._indeterminate = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)

    def _update_progress(self, fraction, duration):
        self.progress.set(fraction)
        percent = int(fraction * 100)
        if duration:
            elapsed = format_clock(fraction * duration)
            total = format_clock(duration)
            self._set_status(f"Transcribing... {percent}%  ({elapsed} / {total})")
        else:
            self._set_status(f"Transcribing... {percent}%")

    def _on_success(self, output_path, count):
        self.progress.set(1.0)
        self.last_output_dir = output_path.parent
        self.open_button.configure(state="normal")
        self._set_status(f"Done - {count} subtitles\n{output_path}", COLOR_GPU)

    def _on_error(self, exc):
        self.progress.set(0)
        self._set_status(f"Failed: {exc}", COLOR_CPU)
        messagebox.showerror("Generate Subtitles", f"{type(exc).__name__}: {exc}")


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    SubtitleApp().mainloop()


if __name__ == "__main__":
    main()
