# Cyberpunk 2077 Audio Extraction and Speaker Clustering

This repository contains Python scripts designed to extract audio files from Cyberpunk 2077 game archives and subsequently cluster the extracted audio based on speaker similarity.

## Scripts

### 1. `cp2077_extractor.py`

This script extracts embedded files (primarily `.wem` audio and `.bk2` video) from Cyberpunk 2077 `.archive` files. It identifies files based on known signatures and extracts them into organized subdirectories.

**Features:**

*   Detects archive type based on filename prefix (e.g., `lang_en_voice`, `audio_1_general`).
*   Finds file signatures (e.g., `RIFF` for WEM) within the archive data.
*   Extracts files sequentially into a directory named after the archive, with subdirectories for file types (e.g., `lang_en_voice/wem/`, `lang_en_voice/wav/`).
*   **Optional:** Converts extracted `.wem` files to `.wav` format using the external `vgmstream-cli` tool if it is present and executable in the same directory as the script.

**Dependencies:**

*   Python 3
*   `vgmstream-cli` (external executable, required for WAV conversion). Download from [https://github.com/vgmstream/vgmstream/releases](https://github.com/vgmstream/vgmstream/releases) or package managers and place it in the script's directory. Ensure it's executable (`chmod +x vgmstream-cli` on Linux/macOS).

**Usage:**

```bash
python cp2077_extractor.py <path_to_archive_file.archive> [-o <base_output_directory>]
```

*   `<path_to_archive_file.archive>`: The input Cyberpunk 2077 archive file.
*   `-o <base_output_directory>`: (Optional) The base directory where the output folder (named after the archive) will be created. Defaults to the current directory.

**Example:**

```bash
python cp2077_extractor.py /path/to/cp2077/archives/lang_en_voice.archive -o ./extracted_audio
# Output will be in ./extracted_audio/lang_en_voice/wem/ and potentially ./extracted_audio/lang_en_voice/wav/
```

### 2. `cluster_speakers.py`

This script takes a directory of `.wav` files (like those produced by `cp2077_extractor.py`) and clusters them based on speaker similarity using speaker embeddings generated by a pre-trained NeMo model (`titanet-large`).

**Features:**

*   Loads `.wav` files using `librosa`.
*   Extracts speaker embeddings using the NeMo `titanet-large` model.
*   Can save extracted embeddings to a `.npz` file for faster re-runs.
*   Can load pre-extracted embeddings from a `.npz` file.
*   Uses `MiniBatchKMeans` for memory-efficient clustering of potentially large numbers of embeddings.
*   Includes an **estimation mode** (`--estimate_k`) to help determine a suitable number of clusters (`k`) using Agglomerative Clustering on a sample of the embeddings.
*   Outputs a CSV file mapping each input filename to its assigned cluster ID.

**Dependencies:**

*   Python 3
*   `numpy`
*   `torch` (and `torchaudio`)
*   `librosa`
*   `nemo_toolkit[asr]` (Requires installation, see NeMo documentation: [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo))
*   `scikit-learn`
*   `pandas`
*   `tqdm`

*(See `pyproject.toml` and `uv.lock` for specific dependencies used in this project environment).*

**Usage:**

**A) Extract Embeddings and Cluster:**

```bash
python cluster_speakers.py --input_dir <wav_directory> --output_file <results.csv> -k <num_clusters> [--save_embeddings <embeddings.npz>]
```

**B) Cluster from Saved Embeddings:**

```bash
python cluster_speakers.py --load_embeddings <embeddings.npz> --output_file <results.csv> -k <num_clusters>
```

**C) Estimate Number of Clusters (k):**

```bash
python cluster_speakers.py --load_embeddings <embeddings.npz> --estimate_k --estimate_threshold <distance_threshold>
```

*   `--input_dir <wav_directory>`: Directory containing the `.wav` files to process.
*   `--output_file <results.csv>`: Path to save the clustering results CSV.
*   `-k <num_clusters>`: The target number of speaker clusters for KMeans.
*   `--save_embeddings <embeddings.npz>`: (Optional) Path to save the extracted embeddings.
*   `--load_embeddings <embeddings.npz>`: Path to load pre-extracted embeddings.
*   `--estimate_k`: Flag to run in estimation mode.
*   `--estimate_threshold <distance_threshold>`: Cosine distance threshold for Agglomerative Clustering in estimation mode (e.g., 0.5, 0.6). Required for estimation.
*   `--estimate_sample_size <size>`: (Optional) Number of embeddings to sample for estimation (default: 10000).

**Example Workflow:**

1.  **Extract audio:**
    ```bash
    python cp2077_extractor.py path/to/lang_en_voice.archive -o ./output
    # This creates ./output/lang_en_voice/wem/ and ./output/lang_en_voice/wav/
    ```
2.  **(Optional) Save embeddings:**
    ```bash
    python cluster_speakers.py --input_dir ./output/lang_en_voice/wav/ --save_embeddings embeddings_en_voice.npz
    # This just extracts and saves, doesn't cluster yet. Useful for large datasets.
    ```
3.  **(Optional) Estimate k:**
    ```bash
    python cluster_speakers.py --load_embeddings embeddings_en_voice.npz --estimate_k --estimate_threshold 0.55
    # Note the suggested k value printed to the console.
    ```
4.  **Cluster:**
    ```bash
    # Using extracted files directly:
    python cluster_speakers.py --input_dir ./output/lang_en_voice/wav/ --output_file speaker_clusters_en_voice.csv -k 150
    # Or using saved embeddings and estimated k:
    python cluster_speakers.py --load_embeddings embeddings_en_voice.npz --output_file speaker_clusters_en_voice.csv -k 150
    ```

## Notes

*   Extracting and processing audio files, especially generating embeddings, can be computationally intensive and time-consuming.
*   The embeddings file (`.npz`) can be quite large depending on the number of audio files.
*   Ensure you have sufficient disk space for extracted audio and embeddings.
*   The quality of clustering depends heavily on the quality of the audio and the chosen number of clusters (`k`). Experimentation might be needed.
