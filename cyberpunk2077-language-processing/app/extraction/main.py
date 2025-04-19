from config import ExtractionSettings 
from app.extraction.conversion import convert_wem_to_wav
from app.extraction.extract import extract_wem
from app.extraction.setup import directory_setup

def extract_audio():
    """Sets up directories and validates archive signatures based on config."""
    print("--- Starting Extraction --- ")
    try:
        extraction_settings = ExtractionSettings()
        archive_name = extraction_settings.archive_name
        output_path = extraction_settings.output_path
        signatures = extraction_settings.signatures

        print("Configuration loaded.")

        directory_setup(output_path, archive_name)

        extract_wem()

        convert_wem_to_wav(signatures, archive_name)



    except ImportError as e:
        print(f"Error importing configuration or modules: {e}")
    except FileNotFoundError as e:
        print(f"Error during directory setup or validation (file not found): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Extraction setup and validation finished.")