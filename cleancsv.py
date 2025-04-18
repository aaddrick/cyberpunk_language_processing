import csv
import argparse
import sys

def remove_last_two_columns(input_filename, output_filename):
    """
    Reads a CSV file, removes the last two columns from each row,
    and writes the result to a new CSV file.

    Args:
        input_filename (str): The path to the input CSV file.
        output_filename (str): The path to the output CSV file.
    """
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader, None) # Read header row if it exists
            if header:
                if len(header) >= 2:
                    writer.writerow(header[:-2])
                else:
                    writer.writerow(header) # Write original header if less than 2 columns

            for row in reader:
                # Check if the row has at least 2 columns before slicing
                if len(row) >= 2:
                    # Keep all columns except the last two using slicing
                    new_row = row[:-2]
                    writer.writerow(new_row)
                else:
                    # Handle rows with fewer than 2 columns (write them as is)
                    writer.writerow(row)

        print(f"Successfully processed '{input_filename}' and saved results to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove the last two columns from a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")

    args = parser.parse_args()

    remove_last_two_columns(args.input_file, args.output_file)
