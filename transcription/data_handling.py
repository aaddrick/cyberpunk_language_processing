import os
import pandas as pd
import librosa
from tqdm import tqdm
import logging

def validate_inputs(input_csv, audio_dir):
    """Checks if the input CSV and audio directory exist."""
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found: '{input_csv}'")
        return False
    if not os.path.isdir(audio_dir):
        print(f"Error: Audio directory not found: '{audio_dir}'")
        return False
    return True

def load_cluster_data(input_csv):
    """Loads the cluster data from the specified CSV file."""
    print(f"Loading cluster data from '{input_csv}'...")
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} entries.")
        if 'filename' not in df.columns:
            print(f"Error: CSV file '{input_csv}' must contain a 'filename' column.")
            return None
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def prepare_file_paths(df, audio_dir):
    """
    Identifies existing audio file paths based on the DataFrame and audio directory.
    Returns a list of paths, corresponding indices, and a list of skipped files.
    """
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
        return None, None, skipped_files # Return None for paths and indices

    print(f"Found {len(audio_filepaths_to_process)} audio files corresponding to CSV entries.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files because they were not found.")

    return audio_filepaths_to_process, indices_to_process, skipped_files

def load_audio_batch(batch_paths, batch_indices, target_sr, error_log_file):
    """
    Loads a batch of audio files, handling potential loading errors.
    Returns lists of loaded audio signals, their corresponding original indices,
    and a dictionary mapping indices to durations.
    """
    audio_data_batch = []
    current_batch_indices_loaded = []
    durations_map = {}
    load_errors = {} # Track indices with load errors

    for idx, file_path in zip(batch_indices, batch_paths):
        try:
            # Load audio, ensuring mono and resampling to target_sr
            audio_signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
            duration = librosa.get_duration(y=audio_signal, sr=sr)
            durations_map[idx] = duration
            audio_data_batch.append(audio_signal)
            current_batch_indices_loaded.append(idx)
        except Exception as load_e:
            log_msg = f"ERROR_LOADING: File={os.path.basename(file_path)}, Index={idx}, Error={load_e}"
            print(f"\nWarning: {log_msg}. Skipping transcription for this file.")
            logging.warning(log_msg) # Log the warning
            with open(error_log_file, "a") as f_err:
                f_err.write(log_msg + "\n")
            load_errors[idx] = "<LOAD_ERROR>" # Mark index with error

    return audio_data_batch, current_batch_indices_loaded, durations_map, load_errors