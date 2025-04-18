from config import FlowSettings

def main():
    flow_control = FlowSettings()

    if flow_control.extract:
        return
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