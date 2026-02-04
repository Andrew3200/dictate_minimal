# ui/widgets.py
from textual.widgets import Static, RichLog, ProgressBar
from textual.containers import Vertical
from textual.reactive import reactive
from textual.app import ComposeResult
from rich.text import Text

# Import Phase from parent package
from enum import Enum, auto


class Phase(Enum):
    """Mirror of engine.Phase to avoid circular imports."""
    OFF = auto()
    LISTENING = auto()
    SPEAKING = auto()
    THINKING = auto()
    LOADING = auto()


class StatusHeader(Static):
    """Header widget showing session info, state, and hotkeys."""
    
    session_id: reactive[int] = reactive(0)
    phase: reactive[Phase] = reactive(Phase.LOADING)
    clipboard_mode: reactive[bool] = reactive(False)
    status_text: reactive[str] = reactive("INIT")
    debug: reactive[bool] = reactive(False)
    
    PHASE_STYLES = {
        Phase.OFF: ("OFF      ", "dim white"),
        Phase.LOADING: ("LOADING  ", "yellow"),
        Phase.LISTENING: ("LISTENING", "cyan"),
        Phase.SPEAKING: ("SPEAKING ", "bright_green"),
        Phase.THINKING: ("THINKING ", "turquoise2"),  # between cyan and green
    }
    
    def render(self) -> Text:
        text = Text()
        
        # Line 1: Session + Title
        text.append(f"Session {self.session_id:03d}", style="bold bright_white")
        text.append("  |  ", style="dim")
        text.append("RealtimeSTT Dictation", style="bold")
        text.append("\n")
        
        # Line 2: Hotkeys
        text.append("Ctrl+Alt+D ", style="cyan")
        text.append("toggle", style="dim")
        text.append("  |  ", style="dim")
        text.append("Ctrl+Alt+C ", style="cyan")
        text.append("clipboard", style="dim")
        text.append("  |  ", style="dim")
        text.append("Ctrl+Alt+Q ", style="cyan")
        text.append("quit", style="dim")
        text.append("\n\n")
        
        # Line 3: State + Mode
        phase_text, phase_style = self.PHASE_STYLES.get(self.phase, ("UNKNOWN  ", "white"))
        text.append("State: ", style="dim")
        text.append(phase_text, style=phase_style)
        text.append(" | ", style="dim")
        
        if self.clipboard_mode:
            text.append("Clipboard Mode", style="yellow bold")
        else:
            text.append("Direct Typing ", style="dim")
        
        text.append("\n")
        
        # Line 4: Status message
        text.append(self.status_text, style="cyan")
        
        return text


class DictationLog(RichLog):
    """Scrollable log for dictation output using Textual's RichLog."""
    
    def __init__(self, **kwargs):
        super().__init__(highlight=False, markup=False, wrap=True, **kwargs)
        self._draft = ""
        self._draft_line_id = None
    
    def add_final(self, text: str) -> None:
        """Add a finalized line to the log."""
        self.write(Text(f"- {text}", style="white"))
    
    def update_draft(self, text: str) -> None:
        """Update the current draft line (mutable, with cursor)."""
        self._draft = text
        # RichLog doesn't support updating last line, so we'll handle draft display
        # in a separate widget or via app-level coordination
    
    def clear_draft(self) -> None:
        """Clear the draft."""
        self._draft = ""


class DraftLine(Static):
    """Widget showing the current draft with blinking cursor."""
    
    draft: reactive[str] = reactive("")
    blink: reactive[bool] = reactive(True)
    active: reactive[bool] = reactive(False)  # whether we're in LISTENING/SPEAKING state
    
    def render(self) -> Text:
        text = Text()
        if self.draft:
            text.append(self.draft, style="green1")
            if self.blink:
                text.append(" \u2588", style="green1 bold")
        elif self.active:
            if self.blink:
                text.append("\u2588", style="green1 dim")
        return text


class VRAMText(Static):
    """Text display for VRAM stats."""
    
    gpu_name: reactive[str] = reactive("")
    vram_app: reactive[float] = reactive(0.0)
    vram_system: reactive[float] = reactive(0.0)
    vram_free: reactive[float] = reactive(0.0)
    vram_total: reactive[float] = reactive(1.0)
    cuda_available: reactive[bool] = reactive(False)
    
    def render(self) -> Text:
        text = Text()
        
        if not self.cuda_available:
            text.append("GPU: (no CUDA)", style="dim red")
            return text
        
        text.append(self.gpu_name or "Unknown GPU", style="bright_white")
        text.append("  |  ", style="dim")
        text.append("App: ", style="dim")
        text.append(f"{self.vram_app:.1f} GB", style="cyan")
        text.append("  |  ", style="dim")
        text.append("System: ", style="dim")
        text.append(f"{self.vram_system:.1f} GB", style="yellow")
        text.append("  |  ", style="dim")
        text.append("Free: ", style="dim")
        text.append(f"{self.vram_free:.1f} / {self.vram_total:.1f} GB", style="bright_white")
        
        return text


class VRAMFooter(Vertical):
    """Footer showing GPU/VRAM stats with progress bar."""
    
    gpu_name: reactive[str] = reactive("")
    vram_app: reactive[float] = reactive(0.0)
    vram_system: reactive[float] = reactive(0.0)
    vram_free: reactive[float] = reactive(0.0)
    vram_total: reactive[float] = reactive(1.0)
    cuda_available: reactive[bool] = reactive(False)
    debug: reactive[bool] = reactive(False)
    
    def compose(self) -> ComposeResult:
        yield ProgressBar(total=100, show_eta=False, id="vram-bar")
        yield VRAMText(id="vram-text")
    
    def on_mount(self) -> None:
        """Initialize the progress bar."""
        self._update_children()
    
    def watch_vram_free(self) -> None:
        self._update_children()
    
    def watch_vram_total(self) -> None:
        self._update_children()
    
    def watch_vram_app(self) -> None:
        self._update_children()
    
    def watch_vram_system(self) -> None:
        self._update_children()
    
    def watch_gpu_name(self) -> None:
        self._update_children()
    
    def watch_cuda_available(self) -> None:
        self._update_children()
    
    def _update_children(self) -> None:
        """Update progress bar and text with current values."""
        try:
            bar = self.query_one("#vram-bar", ProgressBar)
            text = self.query_one("#vram-text", VRAMText)
            
            # Update text widget
            text.gpu_name = self.gpu_name
            text.vram_app = self.vram_app
            text.vram_system = self.vram_system
            text.vram_free = self.vram_free
            text.vram_total = self.vram_total
            text.cuda_available = self.cuda_available
            
            # Update progress bar (shows % used, not free)
            if self.vram_total > 0:
                pct_used = ((self.vram_total - self.vram_free) / self.vram_total) * 100
                bar.progress = pct_used
            else:
                bar.progress = 0
        except Exception:
            pass  # Widget not yet mounted
