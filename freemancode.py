import struct
from pathlib import Path
import math

# Define Freeman direction codes (a to h for 0 to 7 directions)
DIRECTION_CODES = 'abcdefgh'

def read_pot_file(file_path):
    """Read the .pot file and extract character data and strokes."""
    characters = []
    with open(file_path, 'rb') as f:
        while True:
            # Read sample size (2 bytes, little-endian unsigned short)
            sample_size_data = f.read(2)
            if len(sample_size_data) < 2:
                break  # End of file
            sample_size = struct.unpack('<H', sample_size_data)[0]
            if sample_size == 0:
                break  # Invalid sample size

            # Read tag code (4 bytes, GB code in first 2 bytes)
            tag_code_data = f.read(4)
            if len(tag_code_data) < 4:
                break
            gb_code = tag_code_data[0:2]
            try:
                char = gb_code.decode('gb2312', errors='replace')
            except:
                char = '?'  # Fallback for invalid encoding

            # Read stroke number (2 bytes, little-endian unsigned short)
            stroke_num_data = f.read(2)
            if len(stroke_num_data) < 2:
                break
            stroke_num = struct.unpack('<H', stroke_num_data)[0]

            # Read stroke data
            strokes = []
            for _ in range(stroke_num):
                stroke = []
                while True:
                    point_data = f.read(4)
                    if len(point_data) < 4:
                        break
                    x, y = struct.unpack('<hh', point_data)  # Signed shorts
                    if x == -1 and y == 0:  # Stroke end marker
                        break
                    stroke.append((x, y))
                if len(stroke) > 1:  # Only include strokes with multiple points
                    strokes.append(stroke)

            # Read character end marker
            end_marker_data = f.read(4)
            if len(end_marker_data) < 4:
                break
            end_x, end_y = struct.unpack('<hh', end_marker_data)
            if end_x != -1 or end_y != -1:
                continue  # Skip if end marker is invalid

            if strokes:  # Only add characters with valid strokes
                characters.append((char, strokes))
    return characters

def compute_freeman_code(stroke):
    """Compute Freeman chain code for a stroke."""
    code = []
    for i in range(len(stroke) - 1):
        x1, y1 = stroke[i]
        x2, y2 = stroke[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue  # Skip if no movement
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 360) % 360  # Normalize to 0-360
        direction = int(round(angle / 45)) % 8  # Map to 0-7
        code.append(DIRECTION_CODES[direction])
    return ''.join(code)

def process_characters(characters):
    """Process characters and format output with Hanzi and Freeman codes."""
    output = []
    for char, strokes in characters:
        freeman_codes = [compute_freeman_code(stroke) for stroke in strokes]
        freeman_str = '|'.join(freeman_codes)
        output.append(f"{char}\n{freeman_str}")
    return output

def main():
    """Main function to process the .pot file and print results."""
    file_path = Path("/Users/macbookpro/Desktop/Thesis/C001-f.pot")
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        return
    characters = read_pot_file(file_path)
    output = process_characters(characters)
    for line in output:
        print(line)

if __name__ == "__main__":
    main()
