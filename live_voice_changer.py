"""
Live DSP Voice Changer (Python desktop)
Features:
- Live microphone input and speaker/headphone output
- Current age/gender and target age/voice-style controls
- Pitch shift (real-time FFT-bin remapping; prototype quality)
- Timbre/EQ shaping
- Simple noise gate
- Echo, robotic ring-modulation, telephone/radio filter
- Optional recording to WAV

Install:
    python -m pip install numpy scipy sounddevice soundfile

Run:
    python live_voice_changer.py

Safety: speaker output can feed back into the microphone. Start at low volume.
Headphones are strongly recommended for a clean demo.
"""

import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, sosfilt


SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 512
FFT_SIZE = 2048


def clamp(value, low, high):
    return max(low, min(high, value))


class LiveDSP:
    """Lightweight real-time DSP processor. Intended as a student project prototype."""

    def __init__(self, sample_rate, channels=1, block_size=512, fft_size=2048):
        self.sr = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.fft_size = fft_size
        self.hop = block_size
        self.window = np.hanning(fft_size).astype(np.float32)
        self.freq_axis = np.arange(fft_size // 2 + 1, dtype=np.float32)
        self.input_history = np.zeros((fft_size, channels), dtype=np.float32)
        self.output_overlap = np.zeros((fft_size, channels), dtype=np.float32)
        self.norm_overlap = np.zeros(fft_size, dtype=np.float32)
        self.window_sq = self.window * self.window

        self.delay_samples = int(sample_rate * 0.28)
        self.delay_line = np.zeros((self.delay_samples + block_size + 1, channels), dtype=np.float32)
        self.delay_pos = 0

        self.settings_lock = threading.Lock()
        self.settings = {
            "pitch": 0.0,
            "timbre": 0.4,
            "noise_gate": 0.0,
            "echo": 0.0,
            "robot": 0.0,
            "radio": False,
            "profile": "Neutral",
        }

        self.hp_sos = butter(2, 100, btype="highpass", fs=sample_rate, output="sos")
        self.lp_sos = butter(2, min(7000, sample_rate * 0.4), btype="lowpass", fs=sample_rate, output="sos")
        self.hp_zi = np.zeros((self.hp_sos.shape[0], 2, channels), dtype=np.float32)
        self.lp_zi = np.zeros((self.lp_sos.shape[0], 2, channels), dtype=np.float32)

        self.noise_floor = 0.003
        self.recording_lock = threading.Lock()
        self.recording = False
        self.recorded_chunks = []

    def update_settings(self, **kwargs):
        with self.settings_lock:
            self.settings.update(kwargs)

    def get_settings(self):
        with self.settings_lock:
            return dict(self.settings)

    def _pitch_shift_frame(self, frame, semitones):
        if abs(semitones) < 0.05:
            return frame
        factor = 2.0 ** (semitones / 12.0)
        spectrum = np.fft.rfft(frame * self.window[:, None], axis=0)
        source_positions = self.freq_axis / factor
        shifted = np.empty_like(spectrum)
        for ch in range(self.channels):
            shifted[:, ch] = (
                np.interp(source_positions, self.freq_axis, spectrum[:, ch].real, left=0.0, right=0.0)
                + 1j * np.interp(source_positions, self.freq_axis, spectrum[:, ch].imag, left=0.0, right=0.0)
            )
        shifted_frame = np.fft.irfft(shifted, n=self.fft_size, axis=0).astype(np.float32)
        return shifted_frame * self.window[:, None]

    def _timbre_shape(self, x, profile, strength):
        if strength <= 0.001 or profile == "Neutral":
            return x
        # Broad EQ bands approximate tone colour; this is not independent formant shifting.
        cutoff = {
            "Bright / child-like": 1800,
            "Feminine-style": 1500,
            "Masculine-style": 950,
            "Elderly-style": 2400,
        }.get(profile, 1400)
        cutoff = clamp(cutoff, 150, self.sr * 0.42)
        sos_low = butter(2, cutoff, btype="lowpass", fs=self.sr, output="sos")
        sos_high = butter(2, cutoff, btype="highpass", fs=self.sr, output="sos")
        low = sosfilt(sos_low, x, axis=0)
        high = sosfilt(sos_high, x, axis=0)
        if profile == "Bright / child-like":
            result = low * (1.0 - 0.25 * strength) + high * (1.0 + 0.55 * strength)
        elif profile == "Feminine-style":
            result = low * (1.0 - 0.15 * strength) + high * (1.0 + 0.35 * strength)
        elif profile == "Masculine-style":
            result = low * (1.0 + 0.35 * strength) + high * (1.0 - 0.15 * strength)
        else:  # elderly-style
            result = low * (1.0 - 0.08 * strength) + high * (1.0 - 0.25 * strength)
        return result.astype(np.float32)

    def _process_block(self, input_block):
        settings = self.get_settings()
        x = np.asarray(input_block, dtype=np.float32).copy()
        if x.ndim == 1:
            x = x[:, None]

        # Simple noise gate: attenuates quiet blocks; it is not a studio-grade denoiser.
        gate_strength = float(settings["noise_gate"])
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        self.noise_floor = 0.995 * self.noise_floor + 0.005 * min(rms, self.noise_floor * 3 + 0.0005)
        threshold = self.noise_floor * (1.0 + 18.0 * gate_strength)
        if gate_strength > 0 and rms < threshold:
            x *= max(0.08, rms / (threshold + 1e-8))

        # Telephone/radio bandwidth limitation.
        if settings["radio"]:
            x, self.hp_zi = sosfilt(self.hp_sos, x, axis=0, zi=self.hp_zi)
            x, self.lp_zi = sosfilt(self.lp_sos, x, axis=0, zi=self.lp_zi)

        # FFT frame-based pitch transform with overlap-add. This is a lightweight demo
        # and may sound more metallic than a professional phase-vocoder pitch shifter.
        self.input_history = np.concatenate((self.input_history, x), axis=0)[-self.fft_size:]
        frame_out = self._pitch_shift_frame(self.input_history, float(settings["pitch"]))
        self.output_overlap += frame_out
        self.norm_overlap += self.window_sq
        out = self.output_overlap[:self.hop].copy()
        norm = np.maximum(self.norm_overlap[:self.hop], 0.08)
        out /= norm[:, None]
        self.output_overlap = np.concatenate(
            (self.output_overlap[self.hop:], np.zeros((self.hop, self.channels), dtype=np.float32)), axis=0
        )
        self.norm_overlap = np.concatenate(
            (self.norm_overlap[self.hop:], np.zeros(self.hop, dtype=np.float32)), axis=0
        )

        out = self._timbre_shape(out, settings["profile"], float(settings["timbre"]))

        # Echo via a delay line.
        echo = float(settings["echo"])
        if echo > 0:
            delayed = np.zeros_like(out)
            for i in range(len(out)):
                read_pos = (self.delay_pos - self.delay_samples) % len(self.delay_line)
                delayed[i] = self.delay_line[read_pos]
                self.delay_line[self.delay_pos] = out[i] + delayed[i] * 0.25
                self.delay_pos = (self.delay_pos + 1) % len(self.delay_line)
            out = out + echo * delayed
        else:
            for i in range(len(out)):
                self.delay_line[self.delay_pos] = out[i]
                self.delay_pos = (self.delay_pos + 1) % len(self.delay_line)

        # Robotic ring modulation.
        robot = float(settings["robot"])
        if robot > 0:
            start_time = time.perf_counter()
            t = (np.arange(len(out), dtype=np.float32) + start_time * self.sr) / self.sr
            carrier = np.sin(2 * np.pi * 45.0 * t)[:, None]
            out = (1.0 - robot) * out + robot * out * carrier

        # Soft limiter and output gain.
        out = np.tanh(out * 1.3) * 0.85

        with self.recording_lock:
            if self.recording:
                self.recorded_chunks.append(out.copy())

        return out.astype(np.float32)


class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live DSP Voice Changer")
        self.root.geometry("850x760")
        self.root.minsize(760, 680)

        self.processor = LiveDSP(SAMPLE_RATE, CHANNELS, BLOCK_SIZE, FFT_SIZE)
        self.stream = None
        self.running = False
        self.audio_errors = queue.Queue()
        self.status = tk.StringVar(value="Stopped — configure your voice, then click Start")
        self.current_age = tk.IntVar(value=25)
        self.current_gender = tk.StringVar(value="Male")
        self.target_age = tk.IntVar(value=60)
        self.target_gender = tk.StringVar(value="Female")
        self.profile = tk.StringVar(value="Elderly-style")
        self.pitch = tk.DoubleVar(value=-2.0)
        self.timbre = tk.DoubleVar(value=0.5)
        self.noise_gate = tk.DoubleVar(value=0.2)
        self.echo = tk.DoubleVar(value=0.0)
        self.robot = tk.DoubleVar(value=0.0)
        self.radio = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_errors)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Live Age & Gender-Inspired DSP Voice Changer",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(main, text="Live microphone → DSP effects → speakers/headphones",
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 8))

        warning = ("Important: speakers can feed your processed voice back into the microphone and cause loud howling. "
                   "Start with low volume; headphones are recommended.")
        ttk.Label(main, text=warning, wraplength=800, foreground="#9a4d00").pack(anchor="w", pady=(0, 10))

        details = ttk.LabelFrame(main, text="1. Person and target voice", padding=10)
        details.pack(fill="x", pady=4)
        for col in range(4):
            details.columnconfigure(col, weight=1)
        ttk.Label(details, text="Current age").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(details, from_=1, to=120, textvariable=self.current_age, width=8).grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Label(details, text="Current gender").grid(row=0, column=1, sticky="w")
        ttk.Combobox(details, textvariable=self.current_gender, values=["Female", "Male", "Non-binary / prefer not to say"], state="readonly", width=24).grid(row=1, column=1, sticky="w", padx=(0, 8))
        ttk.Label(details, text="Target age").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(details, from_=1, to=120, textvariable=self.target_age, width=8).grid(row=1, column=2, sticky="w", padx=(0, 8))
        ttk.Label(details, text="Target voice style").grid(row=0, column=3, sticky="w")
        ttk.Combobox(details, textvariable=self.target_gender, values=["Female", "Male", "Neutral / androgynous"], state="readonly", width=22).grid(row=1, column=3, sticky="w")

        ttk.Button(details, text="Suggest settings from age/gender", command=self.suggest_settings).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(details, text="Age/gender select approximate sound presets, not a guaranteed biological conversion.",
                  wraplength=800).grid(row=2, column=2, columnspan=2, sticky="w", padx=8, pady=(10, 0))

        controls = ttk.LabelFrame(main, text="2. Live DSP controls", padding=10)
        controls.pack(fill="x", pady=4)
        controls.columnconfigure(1, weight=1)
        self._slider(controls, "Pitch shift (semitones)", self.pitch, -8, 8, 0)
        self._slider(controls, "Timbre / EQ strength", self.timbre, 0, 1, 1)
        self._slider(controls, "Noise gate strength", self.noise_gate, 0, 1, 2)
        self._slider(controls, "Echo amount", self.echo, 0, 0.8, 3)
        self._slider(controls, "Robot effect amount", self.robot, 0, 1, 4)

        ttk.Label(controls, text="Timbre profile").grid(row=5, column=0, sticky="w", pady=4)
        profiles = ["Neutral", "Bright / child-like", "Feminine-style", "Masculine-style", "Elderly-style"]
        profile_box = ttk.Combobox(controls, textvariable=self.profile, values=profiles, state="readonly", width=25)
        profile_box.grid(row=5, column=1, sticky="w", pady=4)
        ttk.Checkbutton(controls, text="Telephone / walkie-talkie filter", variable=self.radio,
                        command=self.sync_settings).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=10)
        self.start_button = ttk.Button(buttons, text="▶ Start live voice changer", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="■ Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.record_button = ttk.Button(buttons, text="● Start recording output", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Save recording…", command=self.save_recording).pack(side="left")

        ttk.Label(main, textvariable=self.status, wraplength=800).pack(anchor="w", pady=4)
        ttk.Label(main, text=f"Audio format: {SAMPLE_RATE} Hz, mono, block size {BLOCK_SIZE} samples (~{1000*BLOCK_SIZE/SAMPLE_RATE:.1f} ms per block).",
                  foreground="#555").pack(anchor="w", pady=(4, 0))
        ttk.Label(main, text="Prototype note: the FFT pitch shifter may sound metallic; EQ is only a formant-inspired tone approximation.",
                  wraplength=800, foreground="#555").pack(anchor="w", pady=(4, 0))

        for var in [self.pitch, self.timbre, self.noise_gate, self.echo, self.robot, self.profile]:
            var.trace_add("write", lambda *_: self.sync_settings())

    def _slider(self, parent, label, variable, low, high, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        scale = ttk.Scale(parent, variable=variable, from_=low, to=high, orient="horizontal")
        scale.grid(row=row, column=1, sticky="ew", padx=10, pady=4)
        value_label = ttk.Label(parent, width=7, textvariable=tk.StringVar(value=""))
        value_label.grid(row=row, column=2, sticky="e")
        def refresh(*_):
            value_label.configure(text=f"{variable.get():.2f}")
        variable.trace_add("write", refresh)
        refresh()

    def suggest_settings(self):
        try:
            current_age = int(self.current_age.get())
            target_age = int(self.target_age.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid age", "Please enter valid ages from 1 to 120.")
            return

        target_gender = self.target_gender.get()
        current_gender = self.current_gender.get()
        if target_age <= 12:
            profile = "Bright / child-like"
        elif target_age >= 60:
            profile = "Elderly-style"
        elif target_gender == "Female":
            profile = "Feminine-style"
        elif target_gender == "Male":
            profile = "Masculine-style"
        else:
            profile = "Neutral"

        # These are intentionally mild starting guesses, not a scientific age model.
        pitch = np.clip((target_age - current_age) / 20.0, -5.0, 5.0)
        if target_gender in ("Female", "Male") and current_gender in ("Female", "Male") and target_gender != current_gender:
            pitch += 2.0 if target_gender == "Female" else -2.0
        self.profile.set(profile)
        self.pitch.set(float(clamp(pitch, -8, 8)))
        self.timbre.set(0.6)
        self.sync_settings()
        self.status.set("Suggested settings applied. Adjust by ear; age/gender presets are approximate.")

    def sync_settings(self):
        self.processor.update_settings(
            pitch=float(self.pitch.get()),
            timbre=float(self.timbre.get()),
            noise_gate=float(self.noise_gate.get()),
            echo=float(self.echo.get()),
            robot=float(self.robot.get()),
            radio=bool(self.radio.get()),
            profile=str(self.profile.get()),
        )

    def start(self):
        if self.running:
            return
        try:
            self.sync_settings()
            self.processor.input_history.fill(0)
            self.processor.output_overlap.fill(0)
            self.processor.norm_overlap.fill(0)
            self.processor.delay_line.fill(0)
            self.processor.delay_pos = 0
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=(CHANNELS, CHANNELS),
                dtype="float32",
                callback=self.audio_callback,
                latency="low",
            )
            self.stream.start()
            self.running = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status.set("LIVE — speak into the selected microphone. Keep output volume low.")
        except Exception as exc:
            self.stream = None
            messagebox.showerror("Could not start audio", f"{exc}\n\nCheck your microphone/speaker permissions and audio devices.")
            self.status.set("Could not start. Check the selected audio devices.")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            try:
                self.audio_errors.put_nowait(str(status))
            except queue.Full:
                pass
        try:
            if indata.shape[1] != CHANNELS:
                block = np.mean(indata, axis=1, keepdims=True)
            else:
                block = indata
            processed = self.processor._process_block(block)
            if processed.shape[0] != frames:
                processed = processed[:frames]
                if len(processed) < frames:
                    processed = np.pad(processed, ((0, frames-len(processed)), (0, 0)))
            outdata[:] = processed
        except Exception as exc:
            outdata.fill(0)
            try:
                self.audio_errors.put_nowait(f"Processing error: {exc}")
            except queue.Full:
                pass

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set("Stopped.")

    def toggle_recording(self):
        with self.processor.recording_lock:
            if not self.processor.recording:
                self.processor.recorded_chunks = []
                self.processor.recording = True
                self.record_button.configure(text="■ Stop recording output")
                self.status.set("Recording transformed output…")
            else:
                self.processor.recording = False
                self.record_button.configure(text="● Start recording output")
                self.status.set("Recording stopped. Click Save recording to export a WAV file.")

    def save_recording(self):
        with self.processor.recording_lock:
            chunks = list(self.processor.recorded_chunks)
        if not chunks:
            messagebox.showinfo("No recording", "Start recording output and speak first.")
            return
        data = np.concatenate(chunks, axis=0)
        filename = filedialog.asksaveasfilename(
            title="Save transformed voice",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="live_transformed_voice.wav",
        )
        if filename:
            try:
                sf.write(filename, data, SAMPLE_RATE)
                self.status.set(f"Saved transformed audio to {filename}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))

    def check_errors(self):
        messages = []
        while not self.audio_errors.empty():
            try:
                messages.append(self.audio_errors.get_nowait())
            except queue.Empty:
                break
        if messages:
            self.status.set("Audio status: " + " | ".join(messages[-2:]))
        self.root.after(200, self.check_errors)

    def close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
