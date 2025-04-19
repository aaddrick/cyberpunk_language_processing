import os
import sys
from app.utility.directory_tools import dir_setup

def directory_setup(output_path, archive_name):
    try:
        output_path = os.path.join(output_path, archive_name)
    except Exception as e:
        print(f"Error processing archive path: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        dir_setup(output_path)
    except OSError as e:
        print(f"Error creating output directory '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Directory setup complete.")