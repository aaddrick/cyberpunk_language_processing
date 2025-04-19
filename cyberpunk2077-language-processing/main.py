from config import FlowSettings
from app.extraction.main import extract_audio

def main():
    flow_control = FlowSettings()

    if flow_control.extract:
        extract_audio()
    if flow_control.embed:
        return
    if flow_control.estimate_cluster:
        return
    if flow_control.cluster:
        return
    if flow_control.transcribe:
        return

if __name__ == "__main__":
    main()