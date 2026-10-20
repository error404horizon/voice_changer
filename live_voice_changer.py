"""
Live DSP Voice Changer  (v2 - rewritten pitch engine)

What's new compared with v1
---------------------------
* Real phase-vocoder pitch shifter with peak phase-locking (no more metallic
  "FFT bin remapping" sound).
* INDEPENDENT pitch and formant control. Formants (vocal-tract resonances) are
  what make a voice sound male / female / child / old. Pitch alone sounds like
  a chipmunk or a monster; pitch + formants sounds like a different person.
* Age / gender presets are computed from approximate real-world f0 and
  vocal-tract-length values instead of arbitrary offsets.
* All filters keep their state between audio blocks (v1 rebuilt them every
  block, which caused clicks and buzzing).
* Smooth noise gate, continuous ring-mod phase, vectorised echo (v1 used a
  Python loop per sample), soft limiter instead of tanh distortion.
* Audio device selection and automatic sample-rate fallback.
* Fixed the slider value labels, which were always blank in v1.

Install:
    python -m pip install numpy scipy sounddevice soundfile

Run:
    python live_voice_changer.py

USE HEADPHONES. Speaker output feeds back into the microphone.
"""

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from scipy.signal import butter, sosfilt

try:
    import sounddevice as sd
    SD_ERROR = None
except Exception as exc:  # PortAudio missing, etc.
    sd = None
    SD_ERROR = exc

try:
    import soundfile as sf
except Exception:
    sf = None


FFT_SIZE = 2048      # analysis window. Needs to be big enough to resolve voice harmonics.
HOP = 512            # also the audio block size (75 % overlap)
TWO_PI = 2.0 * np.pi


def clamp(value, low, high):
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
#  Pitch + formant shifter (phase vocoder with peak phase locking)
# --------------------------------------------------------------------------- #
class PitchFormantShifter:
    """
    Streaming STFT pitch shifter.

    1. Split the spectrum into  envelope (formants)  x  excitation (harmonics)
       using cepstral smoothing.
    2. Shift the excitation by `pitch` semitones using a phase vocoder where
       every spectral peak keeps its own coherent phase and its neighbouring
       bins keep their phase relation to that peak (Laroche & Dolson 1999).
       This is what removes the metallic / phasey artefacts.
    3. Re-apply the envelope, shifted by `formant` semitones.
    """

    def __init__(self, sample_rate, fft_size=FFT_SIZE, hop=HOP):
        self.sr = sample_rate
        self.N = fft_size
        self.H = hop
        self.bins = fft_size // 2 + 1
        self.window = 0.5 - 0.5 * np.cos(TWO_PI * np.arange(fft_size) / fft_size)
        self.out_scale = hop / np.sum(self.window ** 2)  # overlap-add gain compensation
        self.kf = np.arange(self.bins, dtype=np.float64)
        self.ki = np.arange(self.bins)
        self.expected = TWO_PI * hop * self.kf / fft_size
        self.lifter = max(20, int(0.0018 * sample_rate))  # ~1.8 ms: below any voice pitch period
        freqs = self.kf * sample_rate / fft_size
        self.log2f = np.log2(np.maximum(freqs, 60.0) / 1000.0)
        self.reset()

    def reset(self):
        self.in_buf = np.zeros(self.N)
        self.out_acc = np.zeros(self.N)
        self.last_phase = np.zeros(self.bins)
        self.prev_peaks = np.zeros(0, dtype=int)
        self.prev_syn = np.zeros(0)

    @staticmethod
    def _nearest(sorted_vals, query):
        """Index into sorted_vals of the nearest value, for each query."""
        j = np.searchsorted(sorted_vals, query)
        jl = np.clip(j - 1, 0, len(sorted_vals) - 1)
        jr = np.clip(j, 0, len(sorted_vals) - 1)
        use_left = np.abs(query - sorted_vals[jl]) <= np.abs(sorted_vals[jr] - query)
        return np.where(use_left, jl, jr)

    def process_hop(self, hop_in, pitch_semi, formant_semi, tilt_db_oct):
        N, H = self.N, self.H
        self.in_buf[:-H] = self.in_buf[H:]
        self.in_buf[-H:] = hop_in

        # zero-phase framing (fftshift) so bin phases inside a peak are consistent
        spec = np.fft.rfft(np.fft.fftshift(self.in_buf * self.window))
        mag = np.abs(spec)
        phase = np.angle(spec)

        if mag.max() < 1e-6:  # silence: nothing to track
            self.last_phase[:] = 0.0
            self.prev_peaks = np.zeros(0, dtype=int)
            frame = np.zeros(N)
        else:
            # --- true instantaneous frequency of every bin (in bins) ---
            d = phase - self.last_phase - self.expected
            self.last_phase = phase
            d -= TWO_PI * np.round(d / TWO_PI)
            true_bin = self.kf + d * N / (TWO_PI * H)

            # --- spectral envelope (formants) via cepstral liftering ---
            cep = np.fft.irfft(np.log(mag + 1e-9), n=N)
            cep[self.lifter + 1:N - self.lifter] = 0.0
            env = np.exp(np.fft.rfft(cep).real)
            flat = mag / env  # excitation / harmonics only

            # --- peaks of the excitation ---
            inner = flat[2:-2]
            is_peak = ((inner > flat[:-4]) & (inner > flat[1:-3]) &
                       (inner >= flat[3:-1]) & (inner >= flat[4:]))
            peaks = np.nonzero(is_peak)[0] + 2
            if peaks.size == 0:
                peaks = np.array([int(np.argmax(flat))])
            region_peak = peaks[self._nearest(peaks, self.ki)]  # peak that "owns" each bin

            # --- phase accumulation per peak (tracked frame to frame) ---
            factor = 2.0 ** (pitch_semi / 12.0)
            advance = TWO_PI * H / N * true_bin[peaks] * factor
            if self.prev_peaks.size:
                jj = self._nearest(self.prev_peaks, peaks)
                matched = np.abs(self.prev_peaks[jj] - peaks) <= 2
                syn_p = np.where(matched, self.prev_syn[jj] + advance, phase[peaks])
            else:
                syn_p = phase[peaks].copy()
            syn_p = (syn_p + np.pi) % TWO_PI - np.pi
            self.prev_peaks, self.prev_syn = peaks, syn_p

            syn_by_bin = np.zeros(self.bins)
            syn_by_bin[peaks] = syn_p

            # --- build shifted spectrum: target bin j <- source bin j / factor ---
            src_pos = self.kf / factor
            valid = src_pos <= self.bins - 1
            src = np.minimum(np.rint(src_pos).astype(int), self.bins - 1)
            pk = region_peak[src]
            new_phase = syn_by_bin[pk] + (phase[src] - phase[pk])
            new_mag = np.interp(src_pos, self.kf, flat) * valid

            # --- apply (possibly shifted) formant envelope + brightness tilt ---
            ff = 2.0 ** (formant_semi / 12.0)
            env_s = np.interp(self.kf / ff, self.kf, env)
            gain = env_s
            if abs(tilt_db_oct) > 0.01:
                gain = gain * 10.0 ** (np.clip(tilt_db_oct * self.log2f, -18, 18) / 20.0)

            out_spec = new_mag * gain * np.exp(1j * new_phase)
            frame = np.fft.fftshift(np.fft.irfft(out_spec, n=N)) * self.window * self.out_scale

        self.out_acc += frame
        out = self.out_acc[:H].copy()
        self.out_acc[:-H] = self.out_acc[H:]
        self.out_acc[-H:] = 0.0
        return out.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Voice presets from age / gender
# --------------------------------------------------------------------------- #
def voice_traits(age, gender):
    """Approximate (mean f0 in Hz, formant scale vs adult male, tremor 0-1)."""
    male, female = (120.0, 1.00), (210.0, 1.17)
    if gender == "Male":
        adult = male
    elif gender == "Female":
        adult = female
    else:
        adult = (math.sqrt(male[0] * female[0]), math.sqrt(male[1] * female[1]))

    if age <= 12:
        t = clamp((age - 4) / 8.0, 0, 1)
        f0, fs = 300.0 - 50.0 * t, 1.38 - 0.13 * t
    elif age < 18:  # blend child -> adult through puberty
        t = (age - 12) / 6.0
        f0 = (1 - t) * 250.0 + t * adult[0]
        fs = (1 - t) * 1.25 + t * adult[1]
    else:
        f0, fs = adult
    tremor = 0.0
    if age >= 60:
        t = clamp((age - 60) / 15.0, 0, 1)
        if gender == "Male":
            f0 *= 1 + 0.12 * t
        elif gender == "Female":
            f0 *= 1 - 0.12 * t
        fs *= 1 - 0.03 * t
        tremor = 0.6 * t
    return f0, fs, tremor


# --------------------------------------------------------------------------- #
#  Full effect chain
# --------------------------------------------------------------------------- #
class LiveDSP:
    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.shifter = PitchFormantShifter(sample_rate)
        self.in_fifo = np.zeros(0, dtype=np.float32)
        self.out_fifo = np.zeros(0, dtype=np.float32)

        self.settings_lock = threading.Lock()
        self.settings = dict(pitch=0.0, formant=0.0, brightness=0.0, gate_db=-80.0,
                             echo=0.0, robot=0.0, tremor=0.0, radio=False, volume=0.8)

        # stateful filters (state is carried across blocks -> no clicks)
        self.hp_sos = butter(2, 80, btype="highpass", fs=sample_rate, output="sos")
        self.hp_zi = np.zeros((self.hp_sos.shape[0], 2))
        self.bp_sos = butter(3, [300, min(3400, 0.45 * sample_rate)], btype="bandpass",
                             fs=sample_rate, output="sos")
        self.bp_zi = np.zeros((self.bp_sos.shape[0], 2))

        self.gate_gain = 1.0
        self.gate_hold = 0
        self.delay_samples = int(sample_rate * 0.28)
        self.delay_line = np.zeros(self.delay_samples + 16384, dtype=np.float32)
        self.delay_pos = 0
        self.rm_phase = 0.0
        self.trem_phase = 0.0

        self.recording_lock = threading.Lock()
        self.recording = False
        self.recorded_chunks = []

    # -- settings -------------------------------------------------------- #
    def update_settings(self, **kwargs):
        with self.settings_lock:
            self.settings.update(kwargs)

    def get_settings(self):
        with self.settings_lock:
            return dict(self.settings)

    # -- stages ---------------------------------------------------------- #
    def _noise_gate(self, x, thr_db):
        if thr_db <= -79:
            self.gate_gain = 1.0
            return x
        n = len(x)
        level_db = 20 * math.log10(float(np.sqrt(np.mean(x * x))) + 1e-9)
        if level_db > thr_db:
            self.gate_hold = int(0.15 * self.sr)  # hold open 150 ms to avoid chatter
        target = 1.0 if self.gate_hold > 0 else 0.0
        self.gate_hold = max(0, self.gate_hold - n)
        tau = 0.005 if target > self.gate_gain else 0.08
        new_gain = self.gate_gain + (target - self.gate_gain) * (1 - math.exp(-n / (tau * self.sr)))
        ramp = np.linspace(self.gate_gain, new_gain, n, endpoint=False, dtype=np.float32)
        self.gate_gain = new_gain
        return x * ramp

    def process(self, block):
        s = self.get_settings()
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        n = len(x)

        x, self.hp_zi = sosfilt(self.hp_sos, x, zi=self.hp_zi)  # remove rumble / DC
        x = self._noise_gate(x.astype(np.float32), s["gate_db"])

        # --- pitch / formant (hop by hop) ---
        self.in_fifo = np.concatenate((self.in_fifo, x))
        produced = []
        while len(self.in_fifo) >= HOP:
            hop, self.in_fifo = self.in_fifo[:HOP], self.in_fifo[HOP:]
            semis = s["pitch"] + 0.35 * s["tremor"] * math.sin(self.trem_phase)  # aged-voice wobble
            self.trem_phase = (self.trem_phase + TWO_PI * 5.5 * HOP / self.sr) % TWO_PI
            produced.append(self.shifter.process_hop(hop, semis, s["formant"], s["brightness"]))
        if produced:
            self.out_fifo = np.concatenate([self.out_fifo] + produced)

        if len(self.out_fifo) >= n:
            out, self.out_fifo = self.out_fifo[:n], self.out_fifo[n:]
        else:  # underrun (only if the device block size isn't a multiple of HOP)
            out = np.zeros(n, dtype=np.float32)
            out[n - len(self.out_fifo):] = self.out_fifo
            self.out_fifo = np.zeros(0, dtype=np.float32)
        out = out.astype(np.float32)

        if s["tremor"] > 0.001:  # amplitude wobble
            ph = self.trem_phase + TWO_PI * 5.5 * np.arange(n) / self.sr
            out *= (1.0 - 0.18 * s["tremor"] * (0.5 - 0.5 * np.cos(ph))).astype(np.float32)

        if s["radio"]:
            out, self.bp_zi = sosfilt(self.bp_sos, out, zi=self.bp_zi)
            out = (out * 1.6).astype(np.float32)

        # echo (vectorised; delay is longer than a block so reads never hit fresh writes)
        L, D = len(self.delay_line), self.delay_samples
        idx = (self.delay_pos + np.arange(n)) % L
        if s["echo"] > 0.001:
            delayed = self.delay_line[(idx - D) % L]
            self.delay_line[idx] = out + 0.35 * delayed
            out = out + s["echo"] * delayed
        else:
            self.delay_line[idx] = out
        self.delay_pos = (self.delay_pos + n) % L

        if s["robot"] > 0.001:  # ring modulation with continuous phase
            ph = self.rm_phase + TWO_PI * 50.0 * np.arange(n) / self.sr
            out = (1 - s["robot"]) * out + s["robot"] * out * np.sin(ph).astype(np.float32)
            self.rm_phase = (self.rm_phase + TWO_PI * 50.0 * n / self.sr) % TWO_PI

        out = out * s["volume"]
        # soft limiter: transparent below 0.8, smooth saturation above
        a = np.abs(out)
        over = a > 0.8
        if np.any(over):
            out[over] = np.sign(out[over]) * (0.8 + 0.19 * np.tanh((a[over] - 0.8) / 0.19))
        out = out.astype(np.float32)

        with self.recording_lock:
            if self.recording:
                self.recorded_chunks.append(out.copy())
        return out


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
def list_devices(kind):
    """kind = 'input' or 'output' -> ['3: Microphone (USB)', ...]"""
    if sd is None:
        return []
    try:
        return [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices())
                if d[f"max_{kind}_channels"] > 0]
    except Exception:
        return []


def default_device(kind, items):
    try:
        idx = sd.default.device[0 if kind == "input" else 1]
        for item in items:
            if item.startswith(f"{idx}:"):
                return item
    except Exception:
        pass
    return items[0] if items else ""


class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live DSP Voice Changer")
        self.root.geometry("880x860")
        self.root.minsize(780, 760)

        self.processor = None
        self.stream = None
        self.sample_rate = 44100
        self.running = False
        self.audio_errors = queue.Queue()
        self.status = tk.StringVar(value="Stopped. Pick your voices, then click Start (headphones recommended).")

        self.current_age = tk.IntVar(value=25)
        self.current_gender = tk.StringVar(value="Male")
        self.target_age = tk.IntVar(value=25)
        self.target_gender = tk.StringVar(value="Female")
        self.pitch = tk.DoubleVar(value=0.0)
        self.formant = tk.DoubleVar(value=0.0)
        self.brightness = tk.DoubleVar(value=0.0)
        self.gate_db = tk.DoubleVar(value=-80.0)
        self.echo = tk.DoubleVar(value=0.0)
        self.robot = tk.DoubleVar(value=0.0)
        self.tremor = tk.DoubleVar(value=0.0)
        self.volume = tk.DoubleVar(value=0.8)
        self.radio = tk.BooleanVar(value=False)

        self.in_devices = list_devices("input")
        self.out_devices = list_devices("output")
        self.in_dev = tk.StringVar(value=default_device("input", self.in_devices))
        self.out_dev = tk.StringVar(value=default_device("output", self.out_devices))

        self._build_ui()
        self.suggest_settings(announce=False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.check_errors)

    # -- UI -------------------------------------------------------------- #
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Live Age & Gender Voice Changer",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(main, text="Microphone → pitch + formant engine → effects → headphones",
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 8))
        ttk.Label(main, wraplength=820, foreground="#9a4d00",
                  text="Use headphones. Speakers feed the processed voice back into the mic and cause howling.").pack(anchor="w", pady=(0, 8))

        dev = ttk.LabelFrame(main, text="Audio devices", padding=10)
        dev.pack(fill="x", pady=4)
        dev.columnconfigure(1, weight=1)
        ttk.Label(dev, text="Microphone").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(dev, textvariable=self.in_dev, values=self.in_devices, state="readonly").grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(dev, text="Headphones / output").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(dev, textvariable=self.out_dev, values=self.out_devices, state="readonly").grid(row=1, column=1, sticky="ew", padx=8)

        details = ttk.LabelFrame(main, text="1. Who you are → who you want to sound like", padding=10)
        details.pack(fill="x", pady=4)
        for col in range(4):
            details.columnconfigure(col, weight=1)
        ttk.Label(details, text="Your age").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(details, from_=1, to=100, textvariable=self.current_age, width=8).grid(row=1, column=0, sticky="w")
        ttk.Label(details, text="Your voice").grid(row=0, column=1, sticky="w")
        ttk.Combobox(details, textvariable=self.current_gender, values=["Female", "Male", "Neutral"],
                     state="readonly", width=14).grid(row=1, column=1, sticky="w")
        ttk.Label(details, text="Target age").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(details, from_=1, to=100, textvariable=self.target_age, width=8).grid(row=1, column=2, sticky="w")
        ttk.Label(details, text="Target voice").grid(row=0, column=3, sticky="w")
        ttk.Combobox(details, textvariable=self.target_gender, values=["Female", "Male", "Neutral"],
                     state="readonly", width=14).grid(row=1, column=3, sticky="w")
        ttk.Button(details, text="Apply preset for these ages/voices", command=self.suggest_settings).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(details, wraplength=420,
                  text="Presets are averages. Fine-tune pitch and formant by ear while talking.").grid(
            row=2, column=2, columnspan=2, sticky="w", padx=8, pady=(10, 0))

        controls = ttk.LabelFrame(main, text="2. Live controls", padding=10)
        controls.pack(fill="x", pady=4)
        controls.columnconfigure(1, weight=1)
        fmt_semi = "{:+.1f} st"
        self._slider(controls, "Pitch (semitones)", self.pitch, -12, 12, 0, fmt_semi)
        self._slider(controls, "Formants (semitones)", self.formant, -6, 6, 1, fmt_semi)
        self._slider(controls, "Brightness (dB/octave)", self.brightness, -4, 4, 2, "{:+.1f}")
        self._slider(controls, "Age tremor", self.tremor, 0, 1, 3, "{:.2f}")
        self._slider(controls, "Noise gate threshold", self.gate_db, -80, -30, 4,
                     lambda v: "Off" if v <= -79 else f"{v:.0f} dB")
        self._slider(controls, "Echo", self.echo, 0, 0.8, 5, "{:.2f}")
        self._slider(controls, "Robot", self.robot, 0, 1, 6, "{:.2f}")
        self._slider(controls, "Output volume", self.volume, 0, 1.5, 7, "{:.2f}")
        ttk.Checkbutton(controls, text="Telephone / walkie-talkie filter", variable=self.radio,
                        command=self.sync_settings).grid(row=8, column=0, columnspan=3, sticky="w", pady=4)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=10)
        self.start_button = ttk.Button(buttons, text="▶ Start", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="■ Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.record_button = ttk.Button(buttons, text="● Start recording output", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Save recording…", command=self.save_recording).pack(side="left")

        ttk.Label(main, textvariable=self.status, wraplength=820).pack(anchor="w", pady=4)
        ttk.Label(main, wraplength=820, foreground="#555",
                  text="Tip: for a male→female voice raise pitch AND formants together. "
                       "Pitch only = chipmunk, formants only = 'big/small throat' effect.").pack(anchor="w", pady=(4, 0))

        for var in [self.pitch, self.formant, self.brightness, self.gate_db, self.echo,
                    self.robot, self.tremor, self.volume]:
            var.trace_add("write", lambda *_: self.sync_settings())

    def _slider(self, parent, label, variable, low, high, row, fmt):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(parent, variable=variable, from_=low, to=high, orient="horizontal").grid(
            row=row, column=1, sticky="ew", padx=10, pady=3)
        text = tk.StringVar()
        ttk.Label(parent, width=9, textvariable=text, anchor="e").grid(row=row, column=2, sticky="e")

        def refresh(*_):
            v = variable.get()
            text.set(fmt(v) if callable(fmt) else fmt.format(v))
        variable.trace_add("write", refresh)
        refresh()

    # -- presets / settings --------------------------------------------- #
    def suggest_settings(self, announce=True):
        try:
            ca, ta = int(self.current_age.get()), int(self.target_age.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid age", "Please enter valid ages.")
            return
        f0_c, fs_c, _ = voice_traits(ca, self.current_gender.get())
        f0_t, fs_t, tremor = voice_traits(ta, self.target_gender.get())
        self.pitch.set(round(clamp(12 * math.log2(f0_t / f0_c), -12, 12), 1))
        self.formant.set(round(clamp(12 * math.log2(fs_t / fs_c), -6, 6), 1))
        self.tremor.set(round(tremor, 2))
        self.brightness.set(0.0)
        self.sync_settings()
        if announce:
            self.status.set(f"Preset applied: pitch {self.pitch.get():+.1f} st, formants {self.formant.get():+.1f} st. "
                            "Adjust by ear.")

    def sync_settings(self):
        if self.processor is None:
            return
        self.processor.update_settings(
            pitch=float(self.pitch.get()), formant=float(self.formant.get()),
            brightness=float(self.brightness.get()), gate_db=float(self.gate_db.get()),
            echo=float(self.echo.get()), robot=float(self.robot.get()),
            tremor=float(self.tremor.get()), volume=float(self.volume.get()),
            radio=bool(self.radio.get()))

    # -- audio ----------------------------------------------------------- #
    @staticmethod
    def _dev_index(text):
        try:
            return int(text.split(":")[0])
        except (ValueError, AttributeError):
            return None

    def start(self):
        if self.running:
            return
        if sd is None:
            messagebox.showerror("Audio unavailable", f"sounddevice / PortAudio could not be loaded:\n{SD_ERROR}")
            return
        in_idx, out_idx = self._dev_index(self.in_dev.get()), self._dev_index(self.out_dev.get())
        rates = []
        try:
            for idx, kind in ((in_idx, "input"), (out_idx, "output")):
                if idx is not None:
                    rates.append(int(sd.query_devices(idx)["default_samplerate"]))
            out_ch = 2 if sd.query_devices(out_idx)["max_output_channels"] >= 2 else 1
        except Exception:
            out_ch = 1
        rates += [48000, 44100]
        rates = list(dict.fromkeys(rates))

        last_error = None
        for sr in rates:
            try:
                self.processor = LiveDSP(sr)
                self.sync_settings()
                self.stream = sd.Stream(samplerate=sr, blocksize=HOP, device=(in_idx, out_idx),
                                        channels=(1, out_ch), dtype="float32",
                                        latency="low", callback=self.audio_callback)
                self.stream.start()
                self.sample_rate = sr
                self.running = True
                latency_ms = 1000.0 * (FFT_SIZE + HOP) / sr
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal")
                self.status.set(f"LIVE at {sr} Hz (~{latency_ms:.0f} ms processing delay). Keep volume low.")
                return
            except Exception as exc:
                last_error = exc
                self.stream = None
        messagebox.showerror("Could not start audio",
                             f"{last_error}\n\nTry other input/output devices (same host API, e.g. both WASAPI).")
        self.status.set("Could not start. Check the selected audio devices.")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            try:
                self.audio_errors.put_nowait(str(status))
            except queue.Full:
                pass
        try:
            out = self.processor.process(indata[:, 0])
            if len(out) < frames:
                out = np.pad(out, (0, frames - len(out)))
            outdata[:] = out[:frames, None]
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
        if self.processor is None:
            messagebox.showinfo("Not running", "Click Start first, then record.")
            return
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
        if self.processor is None or sf is None:
            messagebox.showinfo("Unavailable", "Nothing recorded yet (or soundfile is not installed).")
            return
        with self.processor.recording_lock:
            chunks = list(self.processor.recorded_chunks)
        if not chunks:
            messagebox.showinfo("No recording", "Start recording output and speak first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save transformed voice", defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")], initialfile="live_transformed_voice.wav")
        if filename:
            try:
                sf.write(filename, np.concatenate(chunks), self.sample_rate)
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
    VoiceChangerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
