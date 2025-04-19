import os
import sys
from config import ExtractionSettings
from .offsets import find_offsets
from ..utility.directory_tools import dir_setup

extraction_settings = ExtractionSettings()

def extract_wem(sig, archive_path, base_output_path): 
	extracted_files = []
	decimals = []
	output_subdir_name = os.path.splitext(sig[1])[1].lstrip('.') or 'extracted'
	actual_output_path = os.path.join(output_path, output_subdir_name)

	output_subdir_name = os.path.splitext(sig[1])[1].lstrip('.') 
	if not output_subdir_name: 
		output_subdir_name = 'wem'
	output_path = os.path.join(base_output_path, output_subdir_name)
	dir_setup(output_path) 

	print("Searching archive...")
	decimals = find_offsets(sig[0], archive_path) 
	print("Found {} files.".format(len(decimals)))
	extracted_files = []
	with open(archive_path, 'rb') as f:
		for num, dec in enumerate(decimals, 1):
			padded_num_str = f"{num:05d}"
			output_filename = padded_num_str + sig[1]
			output_filepath = os.path.join(output_path, output_filename)
			print(f"Extracting file {num} of {len(decimals)}: {os.path.join(output_subdir_name, output_filename)}") 
			try:
				f.seek(dec + len(sig[0]))
				size_bytes = f.read(4)
				if len(size_bytes) < 4:
					print(f"Warning: Could not read size for file at offset {dec}. Skipping.", file=sys.stderr)
					continue
				# The size in RIFF WAVE files includes the 'WAVE' chunk ID and the data chunk header,
				# but the size field itself is *after* 'RIFF' and the size field.
				# For WEM, the size seems to be the total size *including* the RIFF header (8 bytes).
				# Let's assume the size field represents the data *following* the size field.
				# RIFF structure: 'RIFF' (4) + size (4) + 'WAVE' (4) + chunks...
				# The size field usually covers 'WAVE' + chunks. So total size = size + 8
				size = int.from_bytes(size_bytes, 'little') + 8 # Add 8 for 'RIFF' and size field itself
				
				f.seek(dec) # Go back to the start of the RIFF header
				file_data = f.read(size)
				if len(file_data) < size:
					print(f"Warning: Could only read {len(file_data)} bytes out of expected {size} for file at offset {dec}. File might be truncated.", file=sys.stderr)
					# Continue extraction with truncated data if necessary, or skip
					if len(file_data) == 0:
						continue # Skip if no data could be read

				with open(output_filepath, 'wb') as f2:
					f2.write(file_data)
				extracted_files.append(output_filepath)
			except Exception as e:
				print(f"Error extracting file {num} at offset {dec}: {e}", file=sys.stderr)

	return decimals, extracted_files # Return decimals for conversion logic, and extracted paths










