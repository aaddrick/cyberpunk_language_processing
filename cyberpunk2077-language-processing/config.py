from pydantic_settings import BaseSettings

class FlowSettings(BaseSettings):
    extract: bool = True
    embed: bool = True
    estimate_cluster: bool = True
    cluster: bool = True
    transcribe: bool = True

class ExtractionSettings(BaseSettings):
    auth_key: str = 'xxx' 

class EmbeddingSettings(BaseSettings):
    '''
    save_embeddings = True 
    load_embeddings = True
     - Calculates the embeddings the first time, then reuses the 
       saved embeddings

    save_embeddings = True 
    load_embeddings = False
     - Recalulates the embeddings every time, saving them each time

    save_embeddings = False 
    load_embeddings = True
     - If there is no saved embeddings, recalulate each time. 
       Otherwise use the saved embeddings

    save_embeddings = False
    load_embeddings = False
     - Always recalculate embeddings, never save or load them.
    '''

    save_embeddings: bool = True
    load_embeddings: bool = True
    embeddings_path: str = './data/embeddings/' 

class ClusterEstimationSettings(BaseSettings):
    '''
    estimate_k
     - Run in estimation mode: Use Agglomerative Clustering on 
       a sample to estimate k.

    estimate_threshold
     - Distance threshold for Agglomerative Clustering during 
       k estimation.
    '''

    estimate_k: bool = False
    estimate_threshold: float = 0.5

class ClusterSettings(BaseSettings):
    cluster_input_dir: str = 'xxx'         # Directory containing WAV files for embedding extraction
    cluster_output_csv: str = 'xxx'        # Path to save the CSV clustering results.
    num_clusters: int = 1                  # Target number of speaker clusters for K-Means.
    embedding_model: str = 'titanet_large' # Name of the NeMo speaker embedding model.
    kmeans_batch_size: int = 1024          # Batch size for MiniBatchKMeans.
    limit_files: int = None                # Limit embedding extraction to the first N files found.

class TranscriptionSettings(BaseSettings):
    '''
    transcribe
     - Perform transcription after clustering (if done)

    transcription_input_csv
     - Path to the input CSV for transcription
     - Must contain 'filename' column.

    transcription_audio_dir
     - Path to the directory containing audio files for transcription

    transcription_output_csv
     - Path to save the output CSV file with transcripts

    asr_model
     - Name of the NeMo ASR model.

    asr_batch_size
     - Batch size for ASR transcription.
    '''

    transcribe: bool = True
    transcription_input_csv: str = 'xxx'
    transcription_audio_dir: str = 'xxx'
    transcription_output_csv: str = 'xxx'
    asr_model: str = 'stt_en_fastconformer_hybrid_large_pc'
    asr_batch_size: int = 8