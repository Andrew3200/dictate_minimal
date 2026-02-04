import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import os, site, time, threading, logging
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from multiprocessing import freeze_support
from pynput import keyboard
import pyautogui
from RealtimeSTT import AudioToTextRecorder

# ---- Make cuDNN DLLs visible (venv-local, if present) ----
for sp in site.getsitepackages():
    cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
    if os.path.isdir(cudnn_bin):
        os.add_dll_directory(cudnn_bin)
        break

# ---- logging: keep it quiet ----
logging.basicConfig(level=logging.ERROR)
for name in ("RealtimeSTT", "ctranslate2", "faster_whisper"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ---- keys ----
VK_D = 68
VK_Q = 81

def is_ctrl(k): return k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
def is_alt(k):  return k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
def is_vk(k, code): return isinstance(k, keyboard.KeyCode) and k.vk == code

# ---- state ----
dictation_on = False
pressed = set()
hotkey_lock = False
stop_event = threading.Event()

recorder = None
listener = None

def safe_type(text: str):
    text = (text or "").strip()
    if text:
        pyautogui.typewrite(text + " ")

def set_status(on: bool):
    global dictation_on
    dictation_on = on
    print("[DICTATION]", "ON" if on else "OFF")

def toggle_dictation():
    set_status(not dictation_on)

def quit_app():
    print("\n[EXIT] Shutting down...")
    stop_event.set()
    try:
        if recorder:
            recorder.shutdown()
    except Exception:
        pass
    try:
        if listener:
            listener.stop()
    except Exception:
        pass

def loop_transcribe():
    # Only pull text while dictation is ON
    while not stop_event.is_set():
        if not dictation_on:
            time.sleep(0.05)
            continue
        try:
            txt = recorder.text()     # phrase-chunked by VAD
            if dictation_on:
                safe_type(txt)
        except Exception:
            time.sleep(0.1)

def on_press(key):
    global hotkey_lock
    pressed.add(key)

    ctrl_down = any(is_ctrl(k) for k in pressed)
    alt_down  = any(is_alt(k) for k in pressed)
    if not (ctrl_down and alt_down):
        return

    if hotkey_lock:
        return

    if any(is_vk(k, VK_D) for k in pressed):
        hotkey_lock = True
        toggle_dictation()
        return

    if any(is_vk(k, VK_Q) for k in pressed):
        hotkey_lock = True
        quit_app()
        return

def on_release(key):
    global hotkey_lock
    pressed.discard(key)
    # unlock once ctrl or alt is released
    if not any(is_ctrl(k) for k in pressed) or not any(is_alt(k) for k in pressed):
        hotkey_lock = False

def main():
    global recorder, listener

    freeze_support()

    print("[INIT] Loading model to GPU (warm-up)...")
    recorder = AudioToTextRecorder(
        device="cuda",
        model="small",
        language="en",
        spinner=False,

        # IMPORTANT: keep realtime off for now (you don't need it for phrase dictation)
        enable_realtime_transcription=False,
    )

    print("[READY] Hotkeys:")
    print("  Ctrl+Alt+D  toggle dictation ON/OFF")
    print("  Ctrl+Alt+Q  quit")
    print("Focus any textbox, toggle ON, speak.")

    # start worker thread
    t = threading.Thread(target=loop_transcribe, daemon=True)
    t.start()

    # start hotkeys
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # idle loop (no CPU burn)
    while not stop_event.is_set():
        stop_event.wait(0.2)

if __name__ == "__main__":
    main()
