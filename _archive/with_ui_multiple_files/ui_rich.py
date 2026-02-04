# ui_rich.py
import time
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

class RichUI:
    def __init__(self, engine):
        self.engine = engine
        self.console = Console()

        self.header_lines = [
            "RealtimeSTT Dictation (A-mode: type on final only)",
            "Hotkeys: Ctrl+Alt+D toggle | Ctrl+Alt+Q quit",
        ]
        self.status = "INIT"
        self.draft = ""
        self.history = []   # list[str]
        self.gpu = {}

    def _render(self):
        layout = Layout()

        # header fixed
        header = Text("\n".join(self.header_lines), style="bold")
        header.append("\n")
        header.append(self.status, style="cyan")
        layout.split_column(
            Layout(Panel(header, title="Status", padding=(1, 2)), size=6),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        # body = scrollable-ish (Rich Live redraw; history can grow)
        body_txt = Text()
        body_txt.append("DICTATION OUTPUT (frozen)\n", style="bold green")
        for line in self.history[-200:]:
            body_txt.append(f"- {line}\n")

        body_txt.append("\nDRAFT (rewrites until finalized)\n", style="bold yellow")
        body_txt.append(self.draft or "", style="yellow")

        layout["body"].update(Panel(body_txt, title="Log", padding=(1, 2)))

        # footer status bar
        if self.gpu.get("cuda"):
            foot = f"GPU: {self.gpu.get('gpu')} | VRAM: {self.gpu.get('vram_free_gb')} / {self.gpu.get('vram_total_gb')} GB free"
        else:
            foot = "GPU: (no CUDA)"

        layout["footer"].update(Panel(Align.left(foot), title="Runtime", padding=(0, 2)))
        return layout

    def run(self):
        with Live(self._render(), console=self.console, refresh_per_second=12, screen=False) as live:
            while not self.engine.stop_event.is_set():
                # consume events
                try:
                    kind, payload = self.engine.events.get(timeout=0.05)
                    if kind == "status":
                        self.status = str(payload)
                    elif kind == "draft":
                        self.draft = str(payload)
                    elif kind == "final":
                        txt = str(payload).strip()
                        if txt:
                            self.history.append(txt)
                        self.draft = ""  # finalize clears draft
                    elif kind == "gpu":
                        self.gpu = payload if isinstance(payload, dict) else {}
                except Exception:
                    pass

                live.update(self._render())
                time.sleep(0.01)
