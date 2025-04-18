import pandas as pd
import os
import logging

def initialize_output_csv(output_csv, df_columns):
    """
    Initializes the output CSV file with the specified columns.
    Overwrites the file if it exists.

    Args:
        output_csv (str): Path to the output CSV file.
        df_columns (list): List of column names from the original DataFrame.

    Returns:
        list: The final ordered list of columns for the output CSV, or None on error.
    """
    # Define the desired column order, adding new columns
    cols_order = ['audio_filepath'] + [col for col in df_columns if col not in ['audio_filepath', 'duration', 'transcript']] + ['duration', 'transcript']

    header_df = pd.DataFrame(columns=cols_order)
    try:
        header_df.to_csv(output_csv, index=False, mode='w') # 'w' mode to overwrite/create
        print(f"Output file '{output_csv}' initialized with header.")
        logging.info(f"Initialized output CSV: {output_csv} with columns: {cols_order}")
        return cols_order
    except Exception as e:
        print(f"Error initializing output CSV '{output_csv}': {e}")
        logging.error(f"Failed to initialize output CSV '{output_csv}': {e}")
        return None

def append_batch_to_csv(output_csv, batch_df, cols_order, error_log_file):
    """
    Appends a processed batch DataFrame to the output CSV file.

    Args:
        output_csv (str): Path to the output CSV file.
        batch_df (pd.DataFrame): DataFrame containing the processed batch data.
        cols_order (list): The desired order of columns for saving.
        error_log_file (str): Path to the error log file.

    Returns:
        int: Number of rows successfully saved, or 0 on error.
    """
    try:
        # Ensure only the columns defined in cols_order are present and in the correct order
        # Filter out any columns in batch_df that are not in cols_order
        final_cols = [col for col in cols_order if col in batch_df.columns]
        batch_to_save = batch_df[final_cols]

        batch_to_save.to_csv(output_csv, index=False, mode='a', header=False, na_rep='NaN')
        logging.info(f"Appended {len(batch_to_save)} rows to {output_csv}")
        return len(batch_to_save)
    except Exception as save_e:
        batch_indices = batch_df.index.tolist()
        log_msg = f"ERROR_SAVING_BATCH: Batch indices {batch_indices}, Error={save_e}"
        print(f"\nWarning: {log_msg}. Could not append batch to {output_csv}.")
        logging.error(log_msg)
        with open(error_log_file, "a") as f_err:
            f_err.write(log_msg + "\n")
        return 0

def log_summary(total_entries, processed_count, skipped_files, error_log_file, script_log_file):
    """Prints and logs the final processing summary."""
    print(f"\n--- Processing Complete ---")
    print(f"Attempted to process {total_entries} entries from the input CSV.")
    print(f"Successfully processed and saved {processed_count} entries.") # Removed output_csv path for clarity
    if skipped_files:
         print(f"Skipped {len(skipped_files)} files initially because they were not found.")
         logging.info(f"Skipped {len(skipped_files)} files (not found): {[f['filename'] for f in skipped_files]}")

    error_log_exists = os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 20 # Check if more than just header
    if error_log_exists:
        print(f"Check '{error_log_file}' for any warnings or errors during loading, transcription, or saving.")
    else:
        print("No significant errors or warnings were logged during the process.")

    print(f"Check '{script_log_file}' for detailed script execution logs.")
    logging.info(f"Processing complete. Total attempted: {total_entries}, Successfully saved: {processed_count}.")