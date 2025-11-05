import json
import os
import sys

def convert_and_save_descriptions(input_filename, output_filename="descriptions_output.json"):
    """
    Reads a file containing JSON objects (one per line), extracts the 'description'
    field from each, and saves the result as a single JSON array of strings
    to the specified output file.
    """
    descriptions = []

    # 1. Check if the input file exists
    if not os.path.exists(input_filename):
        print(f"Error: The input file '{input_filename}' was not found.", file=sys.stderr)
        return

    # 2. Process the input file
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue

                try:
                    data = json.loads(clean_line)
                    if "description" in data:
                        descriptions.append(data["description"])

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line. Error: {e}", file=sys.stderr)
                except KeyError:
                    print(f"Warning: Skipping object missing 'description' key.", file=sys.stderr)

    except Exception as e:
        print(f"An error occurred while reading the input file: {e}", file=sys.stderr)
        return

    # 3. Write the JSON array to the output file
    if descriptions:
        try:
            # Format the list of strings as a single JSON array (pretty-printed with indent=2)
            json_output = json.dumps(descriptions, indent=2, ensure_ascii=False)

            with open(output_filename, 'w', encoding='utf-8') as outfile:
                outfile.write(json_output)

            print(f"Conversion complete. Successfully saved {len(descriptions)} descriptions to '{output_filename}'.")

        except Exception as e:
            print(f"An error occurred while writing to the output file '{output_filename}': {e}", file=sys.stderr)
    else:
        print("No descriptions were extracted. Output file not created.", file=sys.stderr)


# Define the file names
INPUT_FILE = "ayurvedic_foods_8000.txt"
OUTPUT_FILE = "descriptions_output.json" # The name of the new file to be created

# Execute the conversion and saving process
convert_and_save_descriptions(INPUT_FILE, OUTPUT_FILE)