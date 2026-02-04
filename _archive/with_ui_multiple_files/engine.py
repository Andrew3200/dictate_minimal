# engine.py
import os, site, time, threading, queue, logging
from multiprocessing import freeze_support

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
import torch


VK_D = 68
VK_Q = 81

def is_ctrl(k): return k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
def is_alt(k):  return k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
def is_vk(k, code): return isinstance(k, keyboard.KeyCode) and k.vk == code


class DictationEngine:
    """
    Emits events:
      ("status", str)
      ("draft", str)          # overwritten in UI (backspace effect)
      ("final", str)          # append to history + typed to target
      ("gpu", dict)           # periodic status
    """
    def __init__(self, model="small", lang="en", device="cuda"):
        self.model = model
        self.lang = lang
        self.device = device

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.dictation_on = False

        self._pressed = set()
        self._hotkey_lock = False

        self.recorder = None
        self._listener = None
        self._threads = []

    # ---- status helpers ----
    def _emit(self, kind, payload):
        self.events.put((kind, payload))

    def _gpu_stats(self):
        # Torch-only stats (works without extra libs)
        if not torch.cuda.is_available():
            return {"cuda": False}
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(dev),
            "vram_free_gb": round(free / (1024**3), 2),
            "vram_total_gb": round(total / (1024**3), 2),
        }

    # ---- typing ----
    def _type_final(self, text: str):
        t = (text or "").strip()
        if not t:
            return
        pyautogui.typewrite(t + " ")

    # ---- hotkeys ----
    def _toggle(self):
        self.dictation_on = not self.dictation_on
        self._emit("status", f"DICTATION {'ON' if self.dictation_on else 'OFF'}")

    def _quit(self):
        self._emit("status", "QUIT")
        self.stop_event.set()

    def _on_press(self, key):
        self._pressed.add(key)

        ctrl_down = any(is_ctrl(k) for k in self._pressed)
        alt_down  = any(is_alt(k) for k in self._pressed)
        if not (ctrl_down and alt_down):
            return

        if self._hotkey_lock:
            return

        if any(is_vk(k, VK_D) for k in self._pressed):
            self._hotkey_lock = True
            self._toggle()
            return

        if any(is_vk(k, VK_Q) for k in self._pressed):
            self._hotkey_lock = True
            self._quit()
            return

    def _on_release(self, key):
        self._pressed.discard(key)
        # unlock once ctrl or alt is released
        if not any(is_ctrl(k) for k in self._pressed) or not any(is_alt(k) for k in self._pressed):
            self._hotkey_lock = False

    # ---- workers ----
    def _worker_gpu_poll(self):
        while not self.stop_event.is_set():
            self._emit("gpu", self._gpu_stats())
            time.sleep(0.5)

    def _worker_finalize_and_type(self):
        """
        This is 'A': only commit+type on finalized chunks.
        Draft line is supplied by realtime callbacks (below).
        """
        while not self.stop_event.is_set():
            if not self.dictation_on:
                time.sleep(0.05)
                continue
            try:
                final_text = self.recorder.text()  # blocks until VAD-final
                if self.dictation_on and final_text:
                    self._emit("final", final_text)
                    self._type_final(final_text)
            except Exception as e:
                self._emit("status", f"ERR finalize: {e}")
                time.sleep(0.1)

    # ---- public ----
    def start(self):
        freeze_support()

        self._emit("status", "INIT: loading model / warming GPU…")

        # Draft comes from realtime updates:
        def on_rt_update(t):  # very twitchy
            if self.dictation_on:
                self._emit("draft", t or "")

        def on_rt_stable(t):  # usually nicer
            if self.dictation_on:
                self._emit("draft", t or "")

        self.recorder = AudioToTextRecorder(
            device=self.device,
            model=self.model,
            language=self.lang,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=on_rt_update,
            on_realtime_transcription_stabilized=on_rt_stable,
            spinner=False,
            level=30,  # WARNING
        )

        self._emit("status", "READY: Ctrl+Alt+D toggle, Ctrl+Alt+Q quit")

        # listener
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        # threads
        for target in (self._worker_finalize_and_type, self._worker_gpu_poll):
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
