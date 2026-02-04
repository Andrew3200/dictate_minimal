# engine.py
import os, site, time, threading, queue, logging, re
from multiprocessing import freeze_support
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add cuDNN DLL dir if using pip nvidia-cudnn-cu12
for sp in site.getsitepackages():
    cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
    if os.path.isdir(cudnn_bin):
        os.add_dll_directory(cudnn_bin)
        break

# keep deps quieter
logging.basicConfig(level=logging.ERROR)
for name in ("RealtimeSTT", "ctranslate2", "faster_whisper"):
    logging.getLogger(name).setLevel(logging.ERROR)

from RealtimeSTT import AudioToTextRecorder
from pynput import keyboard
import pyautogui
import pyperclip
import torch

# pynvml for accurate GPU memory (same as Task Manager)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    model: str = "medium"
    language: str = "en"
    device: str = "cuda"
    debug: bool = False  # show debug info in UI when True
    # Speech end cutoff tuning
    post_speech_silence_duration: float = 0.4    # higher = less eager to finalize
    silero_deactivity_detection: bool = True     # more robust end-of-speech
    webrtc_sensitivity: int = 3                  # 0-3, higher = more sensitive
    silero_sensitivity: float = 0.5              # 0-1, higher = more sensitive
    # Realtime transcription
    enable_realtime_transcription: bool = True
    realtime_processing_pause: float = 0.15
    allowed_latency_limit: int = 140


# ============================================================
# State Enums
# ============================================================
class Phase(Enum):
    OFF = auto()         # dictation toggled off
    LISTENING = auto()   # VAD listening, no speech yet
    SPEAKING = auto()    # speech detected
    THINKING = auto()    # transcribing / finalizing
    LOADING = auto()     # model loading at startup


# ============================================================
# Session Logger (change #2)
# ============================================================
class SessionLogger:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        self.session_id = 0
        self.path = ""
    
    def open_new_session(self) -> int:
        os.makedirs(self.base_dir, exist_ok=True)
        existing = []
        for fn in os.listdir(self.base_dir):
            m = re.match(r"session_(\d+)\.txt$", fn)
            if m:
                existing.append(int(m.group(1)))
        self.session_id = (max(existing) + 1) if existing else 1
        self.path = os.path.join(self.base_dir, f"session_{self.session_id:03d}.txt")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"Session {self.session_id:03d} — {datetime.now().isoformat(timespec='seconds')}\n")
            f.write("=" * 60 + "\n\n")
        return self.session_id
    
    def append_recording(self, recording_id: int, lines: list) -> None:
        if not self.path:
            return
        text = "\n".join(line.strip() for line in lines if line.strip())
        if not text:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"[Recording {recording_id}] {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(text + "\n\n")


# ============================================================
# Hotkey constants
# ============================================================
VK_D = 68
VK_Q = 81
VK_C = 67

def is_ctrl(k): return k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
def is_alt(k):  return k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
def is_vk(k, code): return isinstance(k, keyboard.KeyCode) and k.vk == code


# ============================================================
# Typing helpers (change #3 - clipboard mode)
# ============================================================
def type_direct(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    pyautogui.typewrite(text + " ")

def type_clipboard(text: str) -> None:
    """Clipboard paste method - more robust in some apps."""
    text = (text or "").strip()
    if not text:
        return
    old = None
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text + " ")
    time.sleep(0.01)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.01)
    if old is not None:
        try:
            pyperclip.copy(old)
        except Exception:
            pass


# ============================================================
# DictationEngine
# ============================================================
class DictationEngine:
    """
    Emits events:
      ("status", str)           # status message
      ("phase", Phase)          # current phase (change #5)
      ("draft", str)            # live transcription (overwrites)
      ("final", str)            # finalized chunk
      ("gpu", dict)             # periodic GPU stats
      ("clear", None)           # clear display on toggle-off (change #1)
    """
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.dictation_on = False
        self.clipboard_mode = False  # (change #3)
        
        # Session tracking (change #2)
        self.logger = SessionLogger()
        self.session_id = 0
        self.recording_id = 0
        self._current_recording_lines = []  # lines for current recording
        
        self._pressed = set()
        self._hotkey_lock = False
        
        self.recorder = None
        self._listener = None
        self._threads = []
        self._model_ready = threading.Event()
        
        # VRAM tracking - baseline before model loads
        self._vram_baseline_free = None
    
    # ---- event helpers ----
    def _emit(self, kind, payload):
        self.events.put((kind, payload))
    
    def _gpu_stats(self):
        # Use pynvml for accurate readings (same as Task Manager)
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                
                total = info.total
                used = info.used
                free = info.free
                
                # Record baseline on first call (before model loads)
                if self._vram_baseline_free is None:
                    self._vram_baseline_free = free
                
                # App usage = how much free VRAM decreased since baseline
                app_used = max(0, self._vram_baseline_free - free)
                
                # System = baseline usage (what was used before our app started)
                system_used = total - self._vram_baseline_free
                
                return {
                    "cuda": True,
                    "gpu": name,
                    "vram_free_gb": round(free / (1024**3), 2),
                    "vram_total_gb": round(total / (1024**3), 2),
                    "vram_app_gb": round(app_used / (1024**3), 2),
                    "vram_system_gb": round(max(0, system_used) / (1024**3), 2),
                }
            except Exception:
                pass  # Fall through to torch method
        
        # Fallback to torch (less accurate but works)
        if not torch.cuda.is_available():
            return {"cuda": False}
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        
        if self._vram_baseline_free is None:
            self._vram_baseline_free = free
        
        app_used = max(0, self._vram_baseline_free - free)
        system_used = total - self._vram_baseline_free
        
        return {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(dev),
            "vram_free_gb": round(free / (1024**3), 2),
            "vram_total_gb": round(total / (1024**3), 2),
            "vram_app_gb": round(app_used / (1024**3), 2),
            "vram_system_gb": round(max(0, system_used) / (1024**3), 2),
        }
    
    # ---- typing ----
    def _type_final(self, text: str):
        t = (text or "").strip()
        if not t:
            return
        if self.clipboard_mode:
            type_clipboard(t)
        else:
            type_direct(t)
    
    # ---- hotkeys ----
    def _toggle(self):
        was_on = self.dictation_on
        self.dictation_on = not self.dictation_on
        
        if self.dictation_on:
            # Turning ON: start new recording
            self.recording_id += 1
            self._current_recording_lines = []
            self._emit("phase", Phase.LISTENING)
            self._emit("status", f"DICTATION ON (Recording {self.recording_id})")
        else:
            # Turning OFF: save recording to disk, clear display (change #1)
            if self._current_recording_lines:
                self.logger.append_recording(self.recording_id, self._current_recording_lines)
            self._emit("phase", Phase.OFF)
            self._emit("clear", None)  # clear display but already logged
            self._emit("status", "DICTATION OFF")
    
    def _toggle_clipboard(self):
        self.clipboard_mode = not self.clipboard_mode
        mode_str = "Clipboard" if self.clipboard_mode else "Direct"
        self._emit("status", f"Typing mode: {mode_str}")
    
    def _quit(self):
        # Save any pending recording before quit
        if self.dictation_on and self._current_recording_lines:
            self.logger.append_recording(self.recording_id, self._current_recording_lines)
        self._emit("status", "QUIT")
        self.stop_event.set()
    
    def _on_press(self, key):
        self._pressed.add(key)
        
        ctrl_down = any(is_ctrl(k) for k in self._pressed)
        alt_down = any(is_alt(k) for k in self._pressed)
        if not (ctrl_down and alt_down):
            return
        
        if self._hotkey_lock:
            return
        
        if any(is_vk(k, VK_D) for k in self._pressed):
            self._hotkey_lock = True
            self._toggle()
            return
        
        if any(is_vk(k, VK_C) for k in self._pressed):
            self._hotkey_lock = True
            self._toggle_clipboard()
            return
        
        if any(is_vk(k, VK_Q) for k in self._pressed):
            self._hotkey_lock = True
            self._quit()
            return
    
    def _on_release(self, key):
        self._pressed.discard(key)
        if not any(is_ctrl(k) for k in self._pressed) or not any(is_alt(k) for k in self._pressed):
            self._hotkey_lock = False
    
    # ---- workers ----
    def _worker_gpu_poll(self):
        while not self.stop_event.is_set():
            self._emit("gpu", self._gpu_stats())
            time.sleep(0.5)
    
    def _worker_model_load(self):
        """Load model in background."""
        cfg = self.config
        self._emit("phase", Phase.LOADING)
        
        # Detailed loading messages
        self._emit("status", f"[1/4] Initializing Whisper model: {cfg.model}")
        self._emit("status", f"[2/4] Device: {cfg.device} | Language: {cfg.language}")
        
        import time as _time
        start_time = _time.time()
        
        self._emit("status", f"[3/4] Loading {cfg.model} model + Silero VAD (this may take 10-30s)...")
        
        # VAD callbacks for state tracking
        # Note: callbacks may receive args depending on RealtimeSTT version, so accept *args
        def on_vad_detect_start(*args):
            if self.dictation_on:
                self._emit("phase", Phase.LISTENING)
        
        def on_vad_start(*args):
            if self.dictation_on:
                self._emit("phase", Phase.SPEAKING)
        
        def on_recording_stop(*args):
            if self.dictation_on:
                self._emit("phase", Phase.THINKING)
        
        def on_transcription_start(*args):
            if self.dictation_on:
                self._emit("phase", Phase.THINKING)
        
        # Realtime draft callback
        def on_rt_update(t):
            if self.dictation_on:
                self._emit("draft", t or "")
        
        self.recorder = AudioToTextRecorder(
            device=cfg.device,
            model=cfg.model,
            language=cfg.language,
            enable_realtime_transcription=cfg.enable_realtime_transcription,
            realtime_processing_pause=cfg.realtime_processing_pause,
            silero_deactivity_detection=cfg.silero_deactivity_detection,
            post_speech_silence_duration=cfg.post_speech_silence_duration,
            webrtc_sensitivity=cfg.webrtc_sensitivity,
            silero_sensitivity=cfg.silero_sensitivity,
            allowed_latency_limit=cfg.allowed_latency_limit,
            spinner=False,
            level=30,  # WARNING
            # VAD callbacks for state
            on_vad_detect_start=on_vad_detect_start,
            on_vad_start=on_vad_start,
            on_recording_stop=on_recording_stop,
            on_transcription_start=on_transcription_start,
            on_realtime_transcription_update=on_rt_update,
            on_realtime_transcription_stabilized=on_rt_update,
        )
        
        elapsed = _time.time() - start_time
        self._emit("status", f"[4/4] Model loaded in {elapsed:.1f}s - Ready!")
        self._emit("phase", Phase.OFF)
        self._model_ready.set()
    
    def _worker_finalize_and_type(self):
        """Pull finalized chunks and type them."""
        # Wait for model to be ready
        self._model_ready.wait()
        
        while not self.stop_event.is_set():
            if not self.dictation_on:
                time.sleep(0.05)
                continue
            try:
                final_text = self.recorder.text()  # blocks until VAD-final
                if self.dictation_on and final_text:
                    self._emit("final", final_text)
                    self._current_recording_lines.append(final_text)
                    self._type_final(final_text)
                    # Back to listening after final
                    self._emit("phase", Phase.LISTENING)
            except Exception as e:
                self._emit("status", f"ERR finalize: {e}")
                time.sleep(0.1)
    
    # ---- public ----
    def start(self):
        freeze_support()
        
        # Open session (change #2)
        self.session_id = self.logger.open_new_session()
        self._emit("status", f"Session {self.session_id:03d} started")
        
        # Start hotkey listener immediately
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        
        # Start workers
        for target in (self._worker_model_load, self._worker_finalize_and_type, self._worker_gpu_poll):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
    
    def shutdown(self):
        self.stop_event.set()
        try:
            if self._listener:
                self._listener.stop()
        except Exception:
            pass
        try:
            if self.recorder:
                self.recorder.shutdown()
        except Exception:
            pass
