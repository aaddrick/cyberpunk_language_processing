import os
import argparse
import pandas as pd
import torch
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
import warnings
import librosa 
import logging

logging.basicConfig(level=logging.INFO, filename='script_run.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings("ignore", category=FutureWarning)



def main(input_csv, audio_dir, output_csv, asr_model_name="stt_en_fastconformer_hybrid_large_pc", batch_size=8): 
    """
    Loads cluster data, loads audio as mono, transcribes using NeMo ASR (Hybrid Model), and saves results incrementally.
    """
    error_log_file = "transcription_errors.log"
    
    if os.path.exists(error_log_file):
        os.remove(error_log_file)

    with open(error_log_file, 'a') as f:
        f.write("--- Log Start ---\n")
    print(f"Logging transcription errors to: {error_log_file}")

    # --- Input Validation ---
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found: '{input_csv}'")
        return
    if not os.path.isdir(audio_dir):
        print(f"Error: Audio directory not found: '{audio_dir}'")
        return

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load ASR Model ---
    print(f"Loading NeMo ASR Hybrid model ({asr_model_name})...")
    try:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=asr_model_name).to(device)
        asr_model.eval()
        print("ASR model loaded successfully.")
    except Exception as e:
        print(f"\nError loading NeMo ASR model '{asr_model_name}': {e}")
        print("\nAttempting to list available models...") 
        available_models_str = "Available models could not be listed."
        try:
            ctc_models = nemo_asr.models.EncDecCTCModel.list_available_models()
            hybrid_models = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.list_available_models() # Keep this check
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
        return

    print(f"Loading cluster data from '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} entries.")
        if 'filename' not in df.columns:
            print(f"Error: CSV file '{input_csv}' must contain a 'filename' column.")
            return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Prepare File Paths ---
    print("Preparing list of audio file paths...")
    audio_filepaths_to_process = []
    indices_to_process = [] 
    skipped_files = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Checking audio files exist"):
        index = row.Index
        filename = row.filename
        filepath = os.path.join(audio_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: Audio file not found for entry '{filename}', skipping.")
            skipped_files.append({'index': index, 'filename': filename, 'reason': 'File not found'})
            continue
        else:
            audio_filepaths_to_process.append(filepath)
            indices_to_process.append(index)

    if not audio_filepaths_to_process:
        print("Error: No audio files found based on the CSV and audio directory.")
        return

    print(f"Found {len(audio_filepaths_to_process)} audio files corresponding to CSV entries.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files because they were not found.")

    df_processed = df.loc[indices_to_process].copy()
    df_processed['audio_filepath'] = audio_filepaths_to_process 

    target_sr = asr_model.preprocessor._cfg['sample_rate']
    print(f"ASR model expects sample rate: {target_sr}")

    df['duration'] = pd.NA 
    df['transcript'] = "" 
    filepath_map = {index: path for index, path in zip(indices_to_process, audio_filepaths_to_process)}
    df['audio_filepath'] = df.index.map(filepath_map)


    # --- Transcribe Audio Files & Write Incrementally ---
    print(f"Transcribing {len(audio_filepaths_to_process)} audio files (batch size: {batch_size}) and writing results incrementally...")

    original_cols = list(df.columns)
    cols_order = ['audio_filepath'] + [col for col in original_cols if col != 'audio_filepath'] + ['duration', 'transcript']
    header_df = pd.DataFrame(columns=cols_order)
    try:
        header_df.to_csv(output_csv, index=False, mode='w') # 'w' mode to overwrite/create
        print(f"Output file '{output_csv}' initialized with header.")
    except Exception as e:
        print(f"Error initializing output CSV '{output_csv}': {e}")
        return 

    processed_count = 0
    durations_map = {} 
    for i in tqdm(range(0, len(audio_filepaths_to_process), batch_size), desc="Loading, Transcribing & Saving"):
        batch_indices = indices_to_process[i:i+batch_size]
        batch_paths = audio_filepaths_to_process[i:i+batch_size]
        audio_data_batch = []
        current_batch_indices_loaded = []

        for idx, file_path in zip(batch_indices, batch_paths):
            try:
                audio_signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
                duration = librosa.get_duration(y=audio_signal, sr=sr)
                durations_map[idx] = duration 
                audio_data_batch.append(audio_signal)
                current_batch_indices_loaded.append(idx) 
            except Exception as load_e:
                log_msg = f"ERROR_LOADING: File={os.path.basename(file_path)}, Error={load_e}"
                print(f"\nWarning: {log_msg}. Skipping transcription for this file.")
                with open(error_log_file, "a") as f_err:
                    f_err.write(log_msg + "\n")
                df.loc[idx, 'duration'] = None
                df.loc[idx, 'transcript'] = "<LOAD_ERROR>"

        audio_data_for_transcription = []
        indices_for_transcription = []
        for idx, audio_signal in zip(current_batch_indices_loaded, audio_data_batch):
             if df.loc[idx, 'transcript'] != "<LOAD_ERROR>":
                 audio_data_for_transcription.append(audio_signal)
                 indices_for_transcription.append(idx)

        if audio_data_for_transcription:
            try:
                transcribed_batch = asr_model.transcribe(audio_data_batch, batch_size=len(audio_data_batch))

                if isinstance(transcribed_batch, tuple) and len(transcribed_batch) > 0:
                    actual_transcripts_batch = transcribed_batch[0]
                elif isinstance(transcribed_batch, list):
                    actual_transcripts_batch = transcribed_batch
                else:
                    raise TypeError(f"Unexpected output type from transcribe: {type(transcribed_batch)}")

                if len(actual_transcripts_batch) != len(audio_data_batch):
                     raise ValueError(f"Transcript count mismatch: Expected {len(audio_data_batch)}, Got {len(actual_transcripts_batch)}")

                for idx, transcript_result in zip(indices_for_transcription, actual_transcripts_batch):
                    raw_filename = df.loc[idx, 'filename'] 
                    logging.info(f"Raw transcript result for index {idx} (file: {raw_filename}): '{transcript_result}'")

                    final_transcript = None
                    warning_logged = False

                    if hasattr(transcript_result, 'text'):
                         final_transcript = transcript_result.text 
                         if final_transcript is None:
                             final_transcript = "<TRANSCRIPT_EXTRACTED_NONE>"
                             log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Extracted None text from Hypothesis."
                             warning_logged = True
                         elif isinstance(final_transcript, str) and not final_transcript.strip():
                             final_transcript = "<TRANSCRIPT_EXTRACTED_EMPTY>"
                             log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Extracted empty/whitespace text from Hypothesis."
                             warning_logged = True
                    elif transcript_result is None:
                        final_transcript = "<TRANSCRIPT_NONE>"
                        log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received None result."
                        warning_logged = True
                    elif isinstance(transcript_result, str) and not transcript_result.strip():
                        final_transcript = "<TRANSCRIPT_EMPTY>"
                        log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received empty/whitespace result string."
                        warning_logged = True
                    elif transcript_result == "<ERROR>":
                        final_transcript = "<ERROR>"
                        log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received '<ERROR>' string directly."
                        warning_logged = True
                    elif isinstance(transcript_result, str):
                         final_transcript = transcript_result
                    else:
                         final_transcript = f"<UNEXPECTED_TYPE:{type(transcript_result).__name__}>"
                         log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received unexpected result type: {type(transcript_result).__name__}."
                         warning_logged = True

                    if warning_logged:
                        print(f"\nWarning: {log_msg}")
                        with open(error_log_file, "a") as f_err:
                            f_err.write(log_msg + "\n")

                    df.loc[idx, 'transcript'] = final_transcript

                    if 'durations_map' in locals() and idx in durations_map:
                         df.loc[idx, 'duration'] = durations_map[idx]
                    elif 'duration' not in df.columns or pd.isna(df.loc[idx, 'duration']): 
                         df.loc[idx, 'duration'] = None


            except Exception as batch_e:
                batch_start_file = os.path.basename(df.loc[indices_for_transcription[0], 'audio_filepath']) if indices_for_transcription else 'N/A'
                log_msg = f"ERROR_BATCH_TRANSCRIPTION: Batch starting with {batch_start_file}, Error={batch_e}"
                print(f"\nWarning: {log_msg}. Files in this batch marked as error.")
                with open(error_log_file, "a") as f_err:
                    f_err.write(log_msg + "\n")
                for idx in indices_for_transcription: 
                    df.loc[idx, 'transcript'] = "<BATCH_TRANSCRIPTION_ERROR>"


        batch_df = df.loc[batch_indices].copy()


        final_cols = [col for col in cols_order if col in batch_df.columns]
        batch_to_save = batch_df[final_cols]

        try:
            batch_to_save.to_csv(output_csv, index=False, mode='a', header=False, na_rep='NaN')
            processed_count += len(batch_to_save)
        except Exception as save_e:
            log_msg = f"ERROR_SAVING_BATCH: Batch starting index {i}, Error={save_e}"
            print(f"\nWarning: {log_msg}. Could not append batch to {output_csv}.")
            with open(error_log_file, "a") as f_err:
                f_err.write(log_msg + "\n")

    # --- Output Messaging ---
    print(f"\n--- Processing Complete ---")
    print(f"Attempted to process {len(df)} entries from the input CSV.")
    print(f"Successfully processed and saved {processed_count} entries to '{output_csv}'.")
    if skipped_files:
         print(f"Skipped {len(skipped_files)} files initially because they were not found.")
    print(f"Check '{error_log_file}' for any warnings or errors during loading, transcription, or saving.")
    print(f"Check '{logging.getLogger().handlers[0].baseFilename}' for detailed script execution logs.") # Get the actual log file name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files listed in a cluster CSV using NeMo ASR.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., speaker_clusters_nemo_k160.csv). Must contain a 'filename' column.")
    parser.add_argument("audio_dir", help="Path to the directory containing the original WAV files (e.g., lang_en_voice/wav).")
    parser.add_argument("output_csv", help="Path to save the output CSV file with transcripts (e.g., clusters_with_transcripts.csv).")
    parser.add_argument("--asr_model", default="stt_en_fastconformer_hybrid_large_pc", help="Name of the NeMo ASR model to use (default: stt_en_fastconformer_hybrid_large_pc). Check available models and classes if needed.") # Updated default
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for ASR transcription (default: 8). Adjust based on GPU memory. Larger models may require smaller batch sizes.") # Updated default and added note

    args = parser.parse_args()

    main(args.input_csv, args.audio_dir, args.output_csv, args.asr_model, args.batch_size)
