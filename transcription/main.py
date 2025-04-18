import argparse
import pandas as pd
import logging
from tqdm import tqdm
import os

# Import functions from our sub-modules
from .config import setup_logging, setup_device, load_asr_model
from .data_handling import validate_inputs, load_cluster_data, prepare_file_paths, load_audio_batch
from .transcriber import transcribe_batch
from .output import initialize_output_csv, append_batch_to_csv, log_summary

def run_transcription(input_csv, audio_dir, output_csv, asr_model_name, batch_size):
    """
    Orchestrates the audio transcription process using sub-modules.
    """
    # --- Setup ---
    error_log_file = setup_logging() # Gets the error log filename
    device = setup_device()
    asr_model = load_asr_model(asr_model_name, device)
    if asr_model is None:
        return # Exit if model loading failed

    # --- Input Validation ---
    if not validate_inputs(input_csv, audio_dir):
        return

    # --- Load Data ---
    df = load_cluster_data(input_csv)
    if df is None:
        return

    # --- Prepare File Paths ---
    audio_filepaths_to_process, indices_to_process, skipped_files = prepare_file_paths(df, audio_dir)
    if audio_filepaths_to_process is None: # Check if prepare_file_paths indicated an error
        log_summary(len(df) if df is not None else 0, 0, skipped_files, error_log_file, logging.getLogger().handlers[0].baseFilename)
        return

    # --- Initialize Output ---
    # Add placeholder columns before initializing output
    df['duration'] = pd.NA
    df['transcript'] = ""
    filepath_map = {index: path for index, path in zip(indices_to_process, audio_filepaths_to_process)}
    df['audio_filepath'] = df.index.map(filepath_map) # Map paths back to the main df

    cols_order = initialize_output_csv(output_csv, list(df.columns))
    if cols_order is None:
        return # Exit if output initialization failed

    # --- Process Batches ---
    target_sr = asr_model.preprocessor._cfg['sample_rate']
    print(f"ASR model expects sample rate: {target_sr}")
    print(f"Transcribing {len(audio_filepaths_to_process)} audio files (batch size: {batch_size}) and writing results incrementally...")

    processed_count = 0
    total_files_to_process = len(audio_filepaths_to_process)

    for i in tqdm(range(0, total_files_to_process, batch_size), desc="Loading, Transcribing & Saving"):
        batch_indices = indices_to_process[i:min(i + batch_size, total_files_to_process)]
        batch_paths = audio_filepaths_to_process[i:min(i + batch_size, total_files_to_process)]

        # Load audio for the current batch
        audio_data_batch, current_batch_indices_loaded, durations_map, load_errors = load_audio_batch(
            batch_paths, batch_indices, target_sr, error_log_file
        )

        # Update DataFrame with load errors and durations for successfully loaded files
        for idx, error_msg in load_errors.items():
            df.loc[idx, 'transcript'] = error_msg
            df.loc[idx, 'duration'] = None # Ensure duration is None for load errors
        for idx, duration in durations_map.items():
             if idx not in load_errors: # Only update duration if no load error
                df.loc[idx, 'duration'] = duration

        # Prepare data for transcription (exclude files with load errors)
        audio_data_for_transcription = [audio for idx, audio in zip(current_batch_indices_loaded, audio_data_batch) if idx not in load_errors]
        indices_for_transcription = [idx for idx in current_batch_indices_loaded if idx not in load_errors]

        # Transcribe the valid audio data in the batch
        transcripts_map = transcribe_batch(
            asr_model, audio_data_for_transcription, indices_for_transcription, df, error_log_file
        )

        # Update DataFrame with transcription results
        for idx, transcript in transcripts_map.items():
            df.loc[idx, 'transcript'] = transcript
            # Ensure duration is set correctly (it might have been set during loading)
            if idx in durations_map:
                 df.loc[idx, 'duration'] = durations_map[idx]
            elif idx not in load_errors: # If not a load error and no duration found, set to None
                 df.loc[idx, 'duration'] = df.loc[idx].get('duration', None) # Keep existing or set None


        # Prepare the batch DataFrame for saving
        # Select all original indices relevant to this batch (including load errors)
        batch_df_to_save = df.loc[batch_indices].copy()

        # Append the processed batch to the CSV
        saved_count = append_batch_to_csv(output_csv, batch_df_to_save, cols_order, error_log_file)
        processed_count += saved_count

    # --- Final Summary ---
    script_log_file = logging.getLogger().handlers[0].baseFilename
    log_summary(len(df), processed_count, skipped_files, error_log_file, script_log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files listed in a cluster CSV using NeMo ASR.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., speaker_clusters.csv). Must contain a 'filename' column.")
    parser.add_argument("audio_dir", help="Path to the directory containing the original audio files.")
    parser.add_argument("output_csv", help="Path to save the output CSV file with transcripts.")
    parser.add_argument("--asr_model", default="stt_en_fastconformer_hybrid_large_pc", help="Name of the NeMo ASR model (default: stt_en_fastconformer_hybrid_large_pc).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for ASR transcription (default: 8).")

    args = parser.parse_args()

    run_transcription(
        input_csv=args.input_csv,
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        asr_model_name=args.asr_model,
        batch_size=args.batch_size
    )