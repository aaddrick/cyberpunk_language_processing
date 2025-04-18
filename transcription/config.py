import os
import torch
import nemo.collections.asr as nemo_asr
import logging
import warnings

def setup_logging(log_file='script_run.log', error_log_file="transcription_errors.log"):
    """Sets up basic logging and prepares the error log file."""
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
    warnings.filterwarnings("ignore", category=FutureWarning)

    if os.path.exists(error_log_file):
        os.remove(error_log_file)
    with open(error_log_file, 'a') as f:
        f.write("--- Log Start ---\n")
    print(f"Logging transcription errors to: {error_log_file}")
    return error_log_file

def setup_device():
    """Determines and returns the appropriate device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_asr_model(model_name, device):
    """Loads the specified NeMo ASR model."""
    print(f"Loading NeMo ASR Hybrid model ({model_name})...")
    try:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name).to(device)
        asr_model.eval()
        print("ASR model loaded successfully.")
        return asr_model
    except Exception as e:
        print(f"\nError loading NeMo ASR model '{model_name}': {e}")
        print("\nAttempting to list available models...")
        available_models_str = "Available models could not be listed."
        try:
            ctc_models = nemo_asr.models.EncDecCTCModel.list_available_models()
            hybrid_models = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.list_available_models()
            available_models_str = "\nAvailable EncDecCTCModels:\n"
            for model_info in ctc_models:
                available_models_str += f"- {model_info.pretrained_model_name}\n"
            available_models_str += "\nAvailable EncDecHybridRNNTCTCBPEModels:\n"
            for model_info in hybrid_models:
                available_models_str += f"- {model_info.pretrained_model_name}\n"
            print(available_models_str)
        except Exception as list_e:
            print(f"Could not retrieve available models: {list_e}")
        print("\nPlease check the model name and your NeMo installation.")
        print("Ensure you are using the correct model class (CTC vs Hybrid) for the chosen model name.")
        return None # Indicate failure