import os
import logging

def transcribe_batch(asr_model, audio_data_batch, indices_for_transcription, df_full, error_log_file):
    """
    Transcribes a batch of audio data using the provided ASR model.

    Args:
        asr_model: The loaded NeMo ASR model.
        audio_data_batch: A list of audio signals (numpy arrays).
        indices_for_transcription: List of original DataFrame indices corresponding to the audio_data_batch.
        df_full: The full DataFrame (used to get filenames for logging).
        error_log_file: Path to the error log file.

    Returns:
        A dictionary mapping original DataFrame indices to their transcript strings or error markers.
    """
    transcripts_map = {}

    if not audio_data_batch:
        return transcripts_map # Return empty if nothing to transcribe

    try:
        # Perform transcription
        transcribed_batch = asr_model.transcribe(audio_data_batch, batch_size=len(audio_data_batch))

        # --- Process Transcription Results ---
        # Handle different possible return types from transcribe()
        if isinstance(transcribed_batch, tuple) and len(transcribed_batch) > 0:
            actual_transcripts_batch = transcribed_batch[0]
        elif isinstance(transcribed_batch, list):
            actual_transcripts_batch = transcribed_batch
        else:
            raise TypeError(f"Unexpected output type from transcribe: {type(transcribed_batch)}")

        # Check for length mismatch
        if len(actual_transcripts_batch) != len(audio_data_batch):
             raise ValueError(f"Transcript count mismatch: Expected {len(audio_data_batch)}, Got {len(actual_transcripts_batch)}")

        # Process each transcript result
        for idx, transcript_result in zip(indices_for_transcription, actual_transcripts_batch):
            raw_filename = df_full.loc[idx, 'filename'] if 'filename' in df_full.columns and idx in df_full.index else f"Index_{idx}"
            logging.info(f"Raw transcript result for index {idx} (file: {raw_filename}): '{transcript_result}'")

            final_transcript = None
            warning_logged = False
            log_msg = ""

            # Extract text, handling various potential issues/formats
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
            elif transcript_result == "<ERROR>": # Check if the model itself returned an error marker
                final_transcript = "<ERROR>"
                log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received '<ERROR>' string directly."
                warning_logged = True
            elif isinstance(transcript_result, str):
                 final_transcript = transcript_result # Good case: got a non-empty string
            else:
                 final_transcript = f"<UNEXPECTED_TYPE:{type(transcript_result).__name__}>"
                 log_msg = f"WARNING_TRANSCRIPT: File={raw_filename}, Index={idx}, Received unexpected result type: {type(transcript_result).__name__}."
                 warning_logged = True

            # Log warnings if any issues were detected
            if warning_logged:
                print(f"\nWarning: {log_msg}")
                logging.warning(log_msg)
                with open(error_log_file, "a") as f_err:
                    f_err.write(log_msg + "\n")

            transcripts_map[idx] = final_transcript

    except Exception as batch_e:
        # Handle errors during the entire batch transcription process
        batch_start_file = os.path.basename(df_full.loc[indices_for_transcription[0], 'audio_filepath']) if indices_for_transcription and 'audio_filepath' in df_full.columns and indices_for_transcription[0] in df_full.index else 'N/A'
        log_msg = f"ERROR_BATCH_TRANSCRIPTION: Batch starting with file corresponding to index {indices_for_transcription[0]} ({batch_start_file}), Error={batch_e}"
        print(f"\nError: {log_msg}. Files in this batch marked as error.")
        logging.error(log_msg)
        with open(error_log_file, "a") as f_err:
            f_err.write(log_msg + "\n")
        # Mark all files in this batch with an error transcript
        for idx in indices_for_transcription:
            transcripts_map[idx] = "<BATCH_TRANSCRIPTION_ERROR>"

    return transcripts_map