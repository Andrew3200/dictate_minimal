import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# run.py
from engine import DictationEngine, Config
from ui.app import DictationApp


def main():
    # Config with medium model (RTX 3060 12GB can handle it)
    cfg = Config(
        model="large-v3", # medium, low all work
        language="en",
        device="cuda",
        debug=False,  # set True to show debug info
        # Speech end cutoff tuning - adjust these to taste:
        post_speech_silence_duration=0.4,
        silero_deactivity_detection=True,
        webrtc_sensitivity=3,
        silero_sensitivity=0.5,
    )
    
    engine = DictationEngine(config=cfg)
    engine.start()
    
    app = DictationApp(engine)
    try:
        app.run()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
