import os
import sys
from app.extraction.signatures import select_signature

def convert_wem_to_wav(signatures, archive_name):
	sig = select_signature(signatures, archive_name)
	needs_conversion = (sig[1] == '.wem')
	can_convert = False
	vgmstream_cli_filename = 'vgmstream-cli'
	# Look for vgmstream-cli next to the script or executable
	script_dir = os.path.dirname(__file__) if not hasattr(sys, 'frozen') else os.path.dirname(sys.executable)
	vgmstream_cli_path = os.path.join(script_dir, vgmstream_cli_filename)
	
	if needs_conversion:
		print("\nChecking for WEM conversion tool (vgmstream-cli)...")

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