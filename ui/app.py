# ui/app.py
from __future__ import annotations

import asyncio
import queue
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Static, Footer
from rich.text import Text

from .widgets import StatusHeader, DictationLog, DraftLine, VRAMFooter, Phase

if TYPE_CHECKING:
    from engine import DictationEngine


class DictationPanel(Vertical):
    """Container for DictationLog + DraftLine with chunk counter in title."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunk_count = 0
    
    def update_title(self, count: int) -> None:
        self._chunk_count = count
        self.border_title = f"Dictation ({count})"


class DictationApp(App):
    """Textual app for RealtimeSTT dictation with real scrolling."""
    
    CSS_PATH = "styles.tcss"
    
    # Override default Ctrl+C to prevent quit confirmation dialog
    # Our hotkeys (Ctrl+Alt+D/C/Q) are handled by pynput globally
    BINDINGS = [
        ("ctrl+c", "noop", ""),  # Disable default Ctrl+C
    ]
    
    def __init__(self, engine: "DictationEngine"):
        super().__init__()
        self.engine = engine
        self._blink_state = True
        self._chunk_count = 0
    
    def action_noop(self) -> None:
        """Do nothing - placeholder for disabled bindings."""
        pass
    
    def action_quit(self) -> None:
        """Override default quit to skip confirmation dialog."""
        self.exit()
    
    def action_request_quit(self) -> None:
        """Override to skip confirmation dialog."""
        self.exit()
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield StatusHeader(id="header")
        with DictationPanel(id="dictation-panel"):
            yield DictationLog(id="log")
            yield DraftLine(id="draft")
        yield VRAMFooter(id="vram")
    
    async def on_mount(self) -> None:
        """Called when app is mounted - start event processing."""
        # Initialize header with session info
        header = self.query_one("#header", StatusHeader)
        header.session_id = self.engine.session_id
        header.debug = self.engine.config.debug
        
        # Start background tasks
        self.run_worker(self._poll_engine_events(), exclusive=False)
        self.run_worker(self._blink_cursor(), exclusive=False)
    
    async def _poll_engine_events(self) -> None:
        """Poll engine events and update UI."""
        header = self.query_one("#header", StatusHeader)
        panel = self.query_one("#dictation-panel", DictationPanel)
        log = self.query_one("#log", DictationLog)
        draft = self.query_one("#draft", DraftLine)
        vram = self.query_one("#vram", VRAMFooter)
        
        # Initialize panel title
        panel.update_title(self._chunk_count)
        
        while not self.engine.stop_event.is_set():
            # Process all pending events
            events_processed = 0
            while events_processed < 50:
                try:
                    kind, payload = self.engine.events.get_nowait()
                    events_processed += 1
                    
                    if kind == "status":
                        header.status_text = str(payload)
                    
                    elif kind == "phase":
                        # Convert engine.Phase to widgets.Phase by name
                        try:
                            ui_phase = Phase[payload.name]
                            header.phase = ui_phase
                            draft.active = ui_phase in (Phase.LISTENING, Phase.SPEAKING)
                        except (KeyError, AttributeError):
                            pass
                    
                    elif kind == "draft":
                        draft.draft = str(payload)
                    
                    elif kind == "final":
                        txt = str(payload).strip()
                        if txt:
                            log.add_final(txt)
                            self._chunk_count += 1
                            panel.update_title(self._chunk_count)
                        draft.draft = ""
                    
                    elif kind == "gpu":
                        gpu = payload if isinstance(payload, dict) else {}
                        vram.cuda_available = gpu.get("cuda", False)
                        vram.gpu_name = gpu.get("gpu", "")
                        vram.vram_app = gpu.get("vram_app_gb", 0.0)
                        vram.vram_system = gpu.get("vram_system_gb", 0.0)
                        vram.vram_free = gpu.get("vram_free_gb", 0.0)
                        vram.vram_total = gpu.get("vram_total_gb", 1.0)
                        vram.debug = self.engine.config.debug
                        
                        # Calculate VRAM percentage and set app-wide CSS class
                        vram_total = gpu.get("vram_total_gb", 1.0)
                        vram_free = gpu.get("vram_free_gb", 0.0)
                        pct_free = (vram_free / vram_total) * 100 if vram_total > 0 else 100
                        
                        # Debug mode: more aggressive thresholds
                        if self.engine.config.debug:
                            if pct_free > 80:
                                level = "normal"
                            elif pct_free > 70:
                                level = "warning"
                            elif pct_free > 60:
                                level = "danger"
                            else:
                                level = "critical"
                        else:
                            # Normal thresholds
                            if pct_free > 50:
                                level = "normal"
                            elif pct_free > 30:
                                level = "warning"
                            elif pct_free > 15:
                                level = "danger"
                            else:
                                level = "critical"
                        
                        # Update screen CSS class
                        self.screen.remove_class("vram-normal", "vram-warning", "vram-danger", "vram-critical")
                        self.screen.add_class(f"vram-{level}")
                    
                    elif kind == "clear":
                        log.clear()
                        draft.draft = ""
                        self._chunk_count = 0
                        panel.update_title(0)
                
                except queue.Empty:
                    break
            
            # Check if we should quit
            if self.engine.stop_event.is_set():
                self.exit()
                break
            
            await asyncio.sleep(0.05)
    
    async def _blink_cursor(self) -> None:
        """Toggle cursor blink state."""
        draft = self.query_one("#draft", DraftLine)
        while not self.engine.stop_event.is_set():
            self._blink_state = not self._blink_state
            draft.blink = self._blink_state
            await asyncio.sleep(0.6)
