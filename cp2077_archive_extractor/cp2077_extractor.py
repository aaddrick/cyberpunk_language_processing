import os
import sys
import argparse
import subprocess
import shutil


def dir_setup(path):
	if not os.path.isdir(path):
		os.makedirs(path)	

def find(sig, archive_path):
	prev = b''
	decimals = []
	with open(archive_path, 'rb') as f:
		if f.read(4) != b'\x52\x44\x41\x52':
			raise Exception('Not a valid archive file.')
		f.seek(0)
		while True:
			concat_pos = 0
			buf = f.read(max(2048 ** 2, len(sig)))
			if not buf:
				break
			concat = prev + buf
			while True:
				concat_pos = concat.find(sig, concat_pos)
				if concat_pos == -1:
					break
				pos = f.tell() + concat_pos - len(concat)
				if sig == b'\x52\x49\x46\x46':
					cur_pos = f.tell()
					f.seek(pos + 16)
					_byte = f.read(1)
					f.seek(cur_pos)
					if _byte != b'\x42':
						concat_pos += len(sig)
						continue
				decimals.append(pos)
				concat_pos += len(sig)
			prev = buf[-len(sig) + 1:]
	return decimals

def main(sig, archive_path, base_output_path): # Renamed output_path to base_output_path for clarity
	# Define and create the specific output directory for this file type (e.g., 'wem')
	output_subdir_name = os.path.splitext(sig[1])[1].lstrip('.') # Get 'wem' from '.wem'
	if not output_subdir_name: # Handle cases with no extension? Default to 'extracted'
		output_subdir_name = 'extracted'
	output_path = os.path.join(base_output_path, output_subdir_name)
	dir_setup(output_path) # Ensure the subdirectory exists

	print("Searching archive...")
	decimals = find(sig[0], archive_path) # Get decimal offsets first
	print("Found {} files.".format(len(decimals)))
	extracted_files = []
	with open(archive_path, 'rb') as f:
		for num, dec in enumerate(decimals, 1):
			# Zero-pad the number to 5 digits
			padded_num_str = f"{num:05d}"
			output_filename = padded_num_str + sig[1] # e.g., 00001.wem
			# Save to the specific subdirectory (e.g., base_output_path/wem/00001.wem)
			output_filepath = os.path.join(output_path, output_filename)
			print(f"Extracting file {num} of {len(decimals)}: {os.path.join(output_subdir_name, output_filename)}") # Show relative path in log
			try:
				# Seek logic remains the same relative to the archive file
				f.seek(dec + len(sig[0]))
				# Read size correctly (assuming size field is 4 bytes after signature)
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

if __name__ == '__main__':
	# Argument parsing
	parser = argparse.ArgumentParser(description='Extract files from Cyberpunk 2077 archives.')
	parser.add_argument('archive_path', help='Path to the input archive file.')
	parser.add_argument('-o', '--output-dir', default='.', help='Path to the base output directory (default: current directory). A subdirectory named after the archive will be created here.')
	args = parser.parse_args()

	signatures = {
		'audio_1_general': (b'\x52\x49\x46\x46', '.wem'),
		'audio_2_soundbanks': (b'\x52\x49\x46\x46', '.wem'),
		'basegame_5_video': (b'\x4b\x42\x32\x6a', '.bk2'),
		'lang_de_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_en_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_es-es_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_fr_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_it_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_ja_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_ko_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_pl_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_pt_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_ru_voice': (b'\x52\x49\x46\x46', '.wem'),
		'lang_zh-cn_voice': (b'\x52\x49\x46\x46', '.wem')
	}

	# Determine archive type and output path
	try:
		archive_filename = os.path.basename(args.archive_path)
		archive_basename_no_ext = os.path.splitext(archive_filename)[0]
		output_path = os.path.join(args.output_dir, archive_basename_no_ext)
	except Exception as e:
		print(f"Error processing archive path: {e}", file=sys.stderr)
		sys.exit(1)


	# Ensure output directory exists
	try:
		dir_setup(output_path)
	except OSError as e:
		print(f"Error creating output directory '{output_path}': {e}", file=sys.stderr)
		sys.exit(1)


	# Get signature based on archive name
	try:
		sig = signatures[archive_basename_no_ext] # Tuple: (binary_signature, extension)
	except KeyError:
		print(f"Error: Unsupported archive type based on filename prefix: '{archive_basename_no_ext}'", file=sys.stderr)
		print("Supported prefixes:", list(signatures.keys()), file=sys.stderr)
		sys.exit(1) # Exit with an error code

	# --- Conversion Setup ---
	needs_conversion = (sig[1] == '.wem')
	can_convert = False
	vgmstream_cli_filename = 'vgmstream-cli'
	# Look for vgmstream-cli next to the script or executable
	script_dir = os.path.dirname(__file__) if not hasattr(sys, 'frozen') else os.path.dirname(sys.executable)
	vgmstream_cli_path = os.path.join(script_dir, vgmstream_cli_filename)


	if needs_conversion:
		print("\nChecking for WEM conversion tool (vgmstream-cli)...")
		# Check if the cli tool exists and is executable
		if os.path.isfile(vgmstream_cli_path) and os.access(vgmstream_cli_path, os.X_OK):
			print(f"Found vgmstream-cli: {os.path.abspath(vgmstream_cli_path)}")
			can_convert = True
		else:
			print(f"Warning: '{vgmstream_cli_filename}' not found or not executable in the script directory.", file=sys.stderr)
			print(f"         Expected location: {os.path.abspath(vgmstream_cli_path)}", file=sys.stderr)
			print("         Cannot convert .wem to .wav.", file=sys.stderr)
			# Provide hint for Linux/macOS users
			if sys.platform != "win32" and os.path.isfile(vgmstream_cli_path):
				print("         Hint: You might need to make it executable using: chmod +x vgmstream-cli", file=sys.stderr)

	# --- Run Extraction ---
	extracted_files = []
	decimals = []
	try:
		print(f"\nStarting extraction for {args.archive_path}...")
		# We need decimals list later if converting
		# Pass the base output path (e.g., audio_1_general) to main
		decimals, extracted_files = main(sig, args.archive_path, output_path)
		# Determine the actual subdir name used inside main based on the extension
		output_subdir_name = os.path.splitext(sig[1])[1].lstrip('.') or 'extracted'
		actual_output_path = os.path.join(output_path, output_subdir_name)
		print(f"\nExtraction complete. {len(extracted_files)} files saved to: {os.path.abspath(actual_output_path)}")

	except FileNotFoundError:
		print(f"Error: Input archive file not found: {args.archive_path}", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"\nAn error occurred during extraction: {e}", file=sys.stderr)
		# Potentially print traceback for debugging
		# import traceback
		# traceback.print_exc()
		sys.exit(1)
	except KeyboardInterrupt:
		print("\nExtraction interrupted by user.")
		sys.exit(1) # Indicate interruption


	# --- Run Conversion (WEM to WAV using vgmstream-cli) ---
	if needs_conversion and can_convert and extracted_files:
		# Define and create the 'wav' subdirectory
		wav_output_path = os.path.join(output_path, 'wav') # output_path is the base (e.g., audio_1_general)
		try:
			dir_setup(wav_output_path) # Ensure the 'wav' subdirectory exists
			print(f"\nStarting WEM to WAV conversion (saving to: {os.path.abspath(wav_output_path)})...")
			converted_count = 0
			conversion_errors = 0

			# Use the returned list of successfully extracted files (now in the 'wem' subdir)
			for wem_filepath in extracted_files: # e.g., audio_1_general/wem/00001.wem
				base_wem_name = os.path.basename(wem_filepath) # e.g., 00001.wem
				# Output filename will be <padded_number>.wav
				# The base name already includes the padding
				wav_filename = os.path.splitext(base_wem_name)[0] + '.wav' # e.g., 00001.wav
				# Save .wav file in the 'wav' subdirectory
				wav_filepath = os.path.join(wav_output_path, wav_filename) # e.g., audio_1_general/wav/00001.wav

				# Display relative paths for clarity
				relative_wem_path = os.path.relpath(wem_filepath, output_path) # e.g., wem/1.wem
				relative_wav_path = os.path.relpath(wav_filepath, output_path) # e.g., wav/1.wav
				print(f"Converting {relative_wem_path} -> {relative_wav_path}...")

				# Command: ./vgmstream-cli input.wem -o output.wav
				cmd = [vgmstream_cli_path, wem_filepath, '-o', wav_filepath]

				try:
					# Run vgmstream-cli, capture output, check for errors
					result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
					# Optional: print vgmstream-cli output if needed for debugging
					# if result.stdout: print(result.stdout)
					# if result.stderr: print(result.stderr, file=sys.stderr) # vgmstream often prints info to stderr
					converted_count += 1
				except FileNotFoundError:
					# This shouldn't happen if the check above passed, but safety first
					print(f"Error: vgmstream-cli command not found during conversion attempt: {vgmstream_cli_path}", file=sys.stderr)
					can_convert = False # Stop trying to convert
					break
				except subprocess.CalledProcessError as e:
					print(f"Error converting {base_wem_name}: vgmstream-cli failed (exit code {e.returncode})", file=sys.stderr)
					print(f"  Command: {' '.join(e.cmd)}", file=sys.stderr)
					# vgmstream often prints errors to stderr, so capture it
					print(f"  Stderr: {e.stderr.strip()}", file=sys.stderr)
					print(f"  Stdout: {e.stdout.strip()}", file=sys.stderr)
					conversion_errors += 1
				except Exception as e:
					print(f"An unexpected error occurred during conversion of {base_wem_name}: {e}", file=sys.stderr)
					conversion_errors += 1

			print(f"\nConversion finished. {converted_count} files converted successfully.")
			if conversion_errors > 0:
				print(f"Encountered errors during conversion for {conversion_errors} files.", file=sys.stderr)

		except OSError as e:
			# Update error message to refer to the correct directory
			print(f"Error creating WAV output directory '{wav_output_path}': {e}", file=sys.stderr)
		except Exception as e:
			print(f"\nAn error occurred during the conversion process: {e}", file=sys.stderr)
		except KeyboardInterrupt:
			print("\nConversion interrupted by user.")
			# Don't exit here, extraction might have been successful

	elif needs_conversion and not can_convert:
		print("\nSkipping WEM to WAV conversion due to missing vgmstream-cli (see warnings above).")
	elif needs_conversion and not extracted_files:
		print("\nNo WEM files were extracted, skipping WAV conversion.")

	# Keep the script window open until user presses Enter
	try:
		input('\nPress Enter to exit.')
	except EOFError: # Handle case where input is piped or unavailable
		pass
