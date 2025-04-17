import os
import argparse
import numpy as np
import torch
import librosa # Import librosa
from nemo.collections.asr.models import EncDecSpeakerLabelModel # Reverted import path as requested
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering # Re-add AgglomerativeClustering for estimation
from tqdm import tqdm
import pandas as pd
import warnings

# Suppress specific warnings from torchaudio if needed
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
warnings.filterwarnings("ignore", category=FutureWarning) # Often from libraries like numpy/pandas with new versions

def get_embedding(audio_path, embedding_model, target_sr=16000):
    """Loads audio with librosa and extracts speaker embedding using NeMo's infer_segment."""
    try:
        # Load audio with librosa: ensure mono, target sample rate
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # Check if audio is long enough (optional, but good practice)
        min_length = target_sr // 4 # e.g., 0.25 seconds
        if len(audio) < min_length:
             print(f"Warning: Skipping short audio file {os.path.basename(audio_path)} ({len(audio)/target_sr:.2f}s)")
             return None

        # Use infer_segment which expects a numpy array (segment)
        # It returns embedding and logits
        emb_tensor, _ = embedding_model.infer_segment(audio)

        # Convert the resulting tensor embedding to numpy array on CPU
        embedding = emb_tensor.squeeze().cpu().numpy()

        return embedding
    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        return None

def main(input_dir, output_file, num_clusters=None, save_embeddings_path=None, load_embeddings_path=None, estimate_k=False, estimate_sample_size=10000, estimate_threshold=0.5): # Added estimation args
    """Main function to extract embeddings and cluster, or estimate k."""

    # --- Argument Validation for Estimation Mode ---
    if estimate_k:
        if not load_embeddings_path:
            print("Error: --load_embeddings is required when using --estimate_k.")
            return
        if estimate_threshold is None: # Check if threshold was actually provided
             print("Error: --estimate_threshold is required when using --estimate_k.")
             return
        print(f"--- Running in Estimation Mode ---")
        print(f"Loading embeddings from: {load_embeddings_path}")
        print(f"Using sample size: {estimate_sample_size}")
        print(f"Using distance threshold: {estimate_threshold}")
    elif num_clusters is None:
         # If not estimating, num_clusters is required for MiniBatchKMeans
         print("Error: -k/--num_clusters is required unless --estimate_k is used.")
         return


    embeddings_array = None
    filenames = None
    extracted_this_run = False # Flag to track if we extracted or loaded

    # --- Load Embeddings if requested ---
    if load_embeddings_path:
        print(f"Attempting to load embeddings from '{load_embeddings_path}'...")
        try:
            with np.load(load_embeddings_path) as data:
                embeddings_array = data['embeddings']
                filenames = data['filenames']
            print(f"Successfully loaded {len(filenames)} embeddings and filenames.")
            if embeddings_array.shape[0] != len(filenames):
                 raise ValueError("Mismatch between number of embeddings and filenames in loaded file.")
        except FileNotFoundError:
            print(f"Error: Embeddings file not found: '{load_embeddings_path}'")
            return
        except Exception as e:
            print(f"Error loading embeddings file: {e}")
            return
    else:
        # --- Proceed with extraction if not loading ---
        extracted_this_run = True
        if not input_dir or not os.path.isdir(input_dir):
             print(f"Error: Input directory '{input_dir}' not found or not specified (required if not loading embeddings).")
             return

        # --- Device Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- Load Model ---
        print("Loading NeMo speaker embedding model (titanet-large)...")
        try:
            # Ensure model is placed on the correct device
            embedding_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large").to(device) # Changed to underscore
            embedding_model.eval() # Set model to evaluation mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading NeMo model: {e}")
            print("Please ensure nemo_toolkit[asr] is installed correctly.")
            return

        # --- Find WAV Files ---
        wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            print(f"Error: No .wav files found in '{input_dir}'.")
            return
        print(f"Found {len(wav_files)} WAV files.")

        # Limit to the first 100 files for testing (adjust as needed)
        limit = 100000 # Set back to a high limit or remove if desired for full runs
        if len(wav_files) > limit:
            print(f"Limiting processing to the first {limit} files found.")
            wav_files = wav_files[:limit]
        elif not wav_files:
            print(f"Error: No .wav files to process after limiting.")
            return

        # --- Extract Embeddings ---
        print(f"Extracting embeddings for {len(wav_files)} files...")
        embeddings = []
        filenames = []
        for wav_path in tqdm(wav_files, desc="Processing files"):
            # Pass only path and model (model is already on device)
            emb = get_embedding(wav_path, embedding_model)
            if emb is not None:
                embeddings.append(emb)
                filenames.append(os.path.basename(wav_path))

        if not embeddings:
            print("Error: No embeddings could be extracted.")
            return

        embeddings_array = np.array(embeddings)
        filenames = np.array(filenames) # Store filenames as numpy array too
        print(f"Successfully extracted {len(embeddings)} embeddings.")
        print(f"Embedding shape: {embeddings_array.shape}")

        # --- Save Embeddings if requested ---
        if save_embeddings_path and extracted_this_run:
            print(f"Saving embeddings and filenames to '{save_embeddings_path}'...")
            try:
                np.savez(save_embeddings_path, embeddings=embeddings_array, filenames=filenames)
                print("Embeddings saved successfully.")
            except Exception as e:
                print(f"Error saving embeddings: {e}")

    # --- Estimate k or Cluster ---
    if estimate_k:
        if embeddings_array is None or filenames is None:
             print("Error: Embeddings must be loaded using --load_embeddings for estimation mode.")
             return

        # --- Estimate k using Agglomerative Clustering on a sample ---
        print(f"\nEstimating k using Agglomerative Clustering on a sample of {estimate_sample_size} embeddings...")
        if estimate_sample_size > len(embeddings_array):
            print(f"Warning: Sample size ({estimate_sample_size}) is larger than loaded embeddings ({len(embeddings_array)}). Using all loaded embeddings.")
            sample_size = len(embeddings_array)
            sample_indices = np.arange(sample_size)
        else:
            sample_size = estimate_sample_size
            # Ensure reproducibility of the sample
            rng = np.random.default_rng(seed=42)
            sample_indices = rng.choice(len(embeddings_array), size=sample_size, replace=False)

        embeddings_sample = embeddings_array[sample_indices]

        try:
            # Use Agglomerative Clustering just for estimation
            agg_clustering = AgglomerativeClustering(
                n_clusters=None, # Determine based on threshold
                metric='cosine',
                linkage='average',
                distance_threshold=estimate_threshold
            ).fit(embeddings_sample)

            estimated_k = agg_clustering.n_clusters_
            print(f"\n--- Estimation Result ---")
            print(f"Agglomerative clustering on {sample_size} samples with threshold {estimate_threshold} found {estimated_k} clusters.")
            print(f"You can use '-k {estimated_k}' (or a similar value) for MiniBatchKMeans on the full dataset.")
            print(f"--- Exiting Estimation Mode ---")
            return # Exit after estimation

        except MemoryError:
             print(f"\nError: Still ran out of memory trying to estimate k with Agglomerative Clustering on {sample_size} samples.")
             print(f"Try reducing --estimate_sample_size.")
             return
        except Exception as e:
             print(f"\nError during k estimation: {e}")
             return

    else:
        # --- Cluster Embeddings using MiniBatchKMeans ---
        if embeddings_array is None or filenames is None:
             print("Error: Embeddings not loaded or extracted. Cannot cluster.")
             return
        if num_clusters is None: # Should be caught by arg parsing, but double-check
             print("Error: Number of clusters (-k) not specified for MiniBatchKMeans.")
             return

        print(f"\nClustering embeddings into {num_clusters} clusters using MiniBatchKMeans...")
        # Using MiniBatchKMeans for memory efficiency
        # batch_size can be tuned, default is 1024 in recent sklearn versions
        # n_init='auto' is recommended for stability
        # compute_labels=True is needed to get the labels directly after fitting
        mbk = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=0, # for reproducibility
            batch_size=1024, # Adjust if needed, default might be okay
            n_init='auto',
            compute_labels=True,
            verbose=1 # Show progress
        ).fit(embeddings_array)

        labels = mbk.labels_
        # Note: MiniBatchKMeans doesn't report the *found* number of clusters like Agglomerative did,
        # because we specified it via n_clusters.
        print(f"Clustering complete. Assigned files to {num_clusters} clusters.")

        # --- Save Results ---
        print(f"Saving results to '{output_file}'...")
        results_df = pd.DataFrame({
            'filename': filenames,
            'cluster_id': labels
        })
        # Sort by cluster ID then filename for easier viewing
        results_df.sort_values(by=['cluster_id', 'filename'], inplace=True)

        try:
            results_df.to_csv(output_file, index=False)
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster WAV files by speaker using embeddings, or estimate k.")
    # Change input_dir and output_file to named optional arguments
    parser.add_argument("--input_dir", default=None, help="Directory containing WAV files (required for extraction).")
    parser.add_argument("--output_file", default=None, help="Path to save the CSV clustering results (required unless estimating k).")
    # Make num_clusters optional if estimating k
    parser.add_argument("-k", "--num_clusters", type=int, default=None, help="Target number of speaker clusters (required unless estimating k).")
    parser.add_argument("--save_embeddings", default=None, help="Path to save extracted embeddings (e.g., embeddings.npz).")
    parser.add_argument("--load_embeddings", default=None, help="Path to load pre-extracted embeddings (e.g., embeddings.npz).")
    # Arguments for estimation mode
    parser.add_argument("--estimate_k", action='store_true', help="Run in estimation mode: Use Agglomerative Clustering on a sample to estimate k.")
    parser.add_argument("--estimate_sample_size", type=int, default=10000, help="Number of embeddings to sample for k estimation (default: 10000).")
    parser.add_argument("--estimate_threshold", type=float, default=None, help="Distance threshold for Agglomerative Clustering during k estimation (required for estimation mode).")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.estimate_k:
        if not args.load_embeddings:
            parser.error("--load_embeddings is required when using --estimate_k.")
        if args.estimate_threshold is None:
             parser.error("--estimate_threshold is required when using --estimate_k.")
        # output_file and num_clusters are not used in estimation mode
        # Set them explicitly to None to avoid potential issues later
        args.output_file = None
        args.num_clusters = None
        args.input_dir = None # Input dir also not needed for estimation
    else:
        # Normal clustering mode validation
        if args.num_clusters is None:
             parser.error("-k/--num_clusters is required unless using --estimate_k.")
        if args.output_file is None:
             parser.error("--output_file is required unless using --estimate_k.")
        if args.load_embeddings is None and args.input_dir is None:
            parser.error("Either --load_embeddings or --input_dir must be specified when not estimating k.")
        if args.load_embeddings and args.input_dir:
            print("Warning: --load_embeddings provided, --input_dir will be ignored for extraction.")
            args.input_dir = None # Ignore input_dir if loading embeddings
        # estimate_threshold is not used in normal mode
        args.estimate_threshold = None # Ensure it's None if not estimating


    main(
        input_dir=args.input_dir, # Use the potentially modified args.input_dir
        output_file=args.output_file,
        num_clusters=args.num_clusters,
        save_embeddings_path=args.save_embeddings,
        load_embeddings_path=args.load_embeddings,
        estimate_k=args.estimate_k,
        estimate_sample_size=args.estimate_sample_size,
        estimate_threshold=args.estimate_threshold
    )