import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# run.py
from engine import DictationEngine
from ui_rich import RichUI

def main():
    eng = DictationEngine(model="small", lang="en", device="cuda")
    eng.start()
    ui = RichUI(eng)
    try:
        ui.run()
    finally:
        eng.shutdown()

if __name__ == "__main__":
    main()
