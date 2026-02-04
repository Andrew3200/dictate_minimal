# Dictate Minimal

A minimal real-time speech-to-text dictation app using Whisper AI. Speaks into your microphone, types into any application.

## Features

- **Real-time transcription** using OpenAI Whisper (via RealtimeSTT)
- **Global hotkeys** - works from any application
- **Live draft preview** - see what's being transcribed before it's finalized
- **Session logging** - all dictations saved to `logs/` folder
- **VRAM monitoring** - visual indicator of GPU memory usage
- **Clipboard mode** - fallback for apps that block direct typing

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~5GB VRAM for large-v3 model (~2.5GB for medium)

## Installation

```bash
git clone https://github.com/Andrew3200/dictate_minimal.git
cd dictate_minimal
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

### Hotkeys

| Hotkey | Action |
|--------|--------|
| `Ctrl+Alt+D` | Toggle dictation on/off |
| `Ctrl+Alt+C` | Toggle clipboard mode |
| `Ctrl+Alt+Q` | Quit |

### Configuration

Edit `run.py` to change settings:

```python
cfg = Config(
    model="large-v3",      # whisper model: tiny, base, small, medium, large-v3
    language="en",         # language code
    debug=False,           # show debug info
    post_speech_silence_duration=0.4,  # seconds before finalizing
)
```

## How It Works

1. **Toggle ON** (`Ctrl+Alt+D`) - starts listening
2. **Speak** - live transcription appears in green
3. **Pause** - text finalizes and types into active window
4. **Toggle OFF** - saves session to log file

## Tech Stack

- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) - Real-time speech-to-text
- [Textual](https://textual.textualize.io/) - Terminal UI framework
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 Whisper implementation
- [pynput](https://github.com/moses-palmer/pynput) - Global hotkey handling
