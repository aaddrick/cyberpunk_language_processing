import sys

def select_signature(signatures, archive_name):
	
	print(f"Validating signature for archive: '{archive_name}'")
	
	try:
		sig = signatures[archive_name]
	except KeyError:
		print(f"Error: Unsupported archive type based on filename prefix: '{archive_name}'", file=sys.stderr)
		print("Supported prefixes:", list(signatures.keys()), file=sys.stderr)
		sys.exit(1) # Exit with an error code
	
	print("Signature validation complete.")
	return sig