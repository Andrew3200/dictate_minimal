# Code Review Notes

## Summary

Rewrote UI from Rich Live to Textual for real scrolling support. All UI tests pass.

## Changes Made

1. **engine.py**
   - Changed `model: str = "small"` to `model: str = "medium"`
   - Added `debug: bool = False` config flag

2. **New ui/ package**
   - `ui/__init__.py` - Package exports
   - `ui/app.py` - Main Textual app (DictationApp)
   - `ui/widgets.py` - StatusHeader, DictationLog, DraftLine, VRAMFooter
   - `ui/styles.tcss` - Textual CSS styling

3. **run.py** - Updated to use new Textual UI

## Tests Created

- `tests/test_ui_only.py` - Tests widgets without requiring RealtimeSTT/CUDA (ALL PASS)
- `tests/test_imports.py` - Full import test (requires RealtimeSTT)

## Potential Issues to Watch

### 1. Hotkey Handling
The engine uses `pynput` for global hotkeys (Ctrl+Alt+D/C/Q). Textual also has key binding support but we're NOT using it since pynput works globally (even when app doesn't have focus). This should work fine but worth testing.

### 2. Phase Enum Duplication
There are TWO Phase enums now:
- `engine.Phase` - The source of truth
- `ui.widgets.Phase` - Mirror for UI (to avoid importing engine in UI)

They're mapped by NAME in `app.py`:
```python
ui_phase = Phase[payload.name]  # Convert engine.Phase to ui.Phase
```

This works but is fragile - if names diverge, it breaks silently.

### 3. RichLog Auto-Scroll
Textual's `RichLog` should auto-scroll to bottom by default. If it doesn't, we may need to call `log.scroll_end()` after writing.

### 4. Worker Thread Safety
The app polls `engine.events` queue from an async worker. This should be thread-safe (queue.Queue is thread-safe) but watch for race conditions.

### 5. Draft Line Separation
The draft line is a separate widget below the log. This means:
- Finalized text goes to DictationLog (scrollable)
- Draft text goes to DraftLine (static at bottom)

This is intentional - draft stays visible at bottom while history scrolls.

### 6. VRAM App Usage
Using `torch.cuda.memory_reserved()` instead of `memory_allocated()`. Reserved shows total memory held by PyTorch's caching allocator, which is closer to "real" usage but may include cached memory not actively used.

## Files That Can Be Deleted

- `ui_rich.py` - Old Rich Live UI (replaced by ui/ package)

## To Test When Running

1. Does the UI render correctly on startup?
2. Does scrolling work when log fills up?
3. Do hotkeys work (Ctrl+Alt+D/C/Q)?
4. Does VRAM display show non-zero app usage after model loads?
5. Does clipboard mode toggle work?
6. Does the blinking cursor animate smoothly?
7. Do finalized lines stay in history?
8. Does toggle-off clear the display?
