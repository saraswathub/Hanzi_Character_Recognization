import struct
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

# Define Freeman direction codes (a to h for 0 to 7 directions)
DIRECTION_CODES = 'abcdefgh'

def read_pot_file(file_path):
    """Read the .pot file and extract character data and strokes."""
    characters = []
    with open(file_path, 'rb') as f:
        while True:
            sample_size_data = f.read(2)
            if len(sample_size_data) < 2:
                break
            sample_size = struct.unpack('<H', sample_size_data)[0]
            if sample_size == 0:
                break

            tag_code_data = f.read(4)
            if len(tag_code_data) < 4:
                break
            gb_code = tag_code_data[0:2]

            # Improved Chinese character decoding
            char = decode_chinese_character(gb_code)

            stroke_num_data = f.read(2)
            if len(stroke_num_data) < 2:
                break
            stroke_num = struct.unpack('<H', stroke_num_data)[0]

            strokes = []
            for _ in range(stroke_num):
                stroke = []
                while True:
                    point_data = f.read(4)
                    if len(point_data) < 4:
                        break
                    x, y = struct.unpack('<hh', point_data)
                    if x == -1 and y == 0:
                        break
                    stroke.append((x, y))
                if len(stroke) > 1:
                    strokes.append(stroke)

            end_marker_data = f.read(4)
            if len(end_marker_data) < 4:
                break
            end_x, end_y = struct.unpack('<hh', end_marker_data)
            if end_x != -1 or end_y != -1:
                continue

            if strokes:
                characters.append((char, strokes))
    return characters

def decode_chinese_character(gb_code):
    """Improved Chinese character decoding with proper GB2312 handling."""
    # First try GB2312 which is the most common for Chinese characters
    try:
        # GB2312 is the standard encoding for simplified Chinese
        char = gb_code.decode('gb2312')
        # Check if it's a valid printable character
        if char and char.isprintable() and not char.isspace():
            return char
    except (UnicodeDecodeError, LookupError):
        pass
    
    # Then try GBK (extended GB2312)
    try:
        char = gb_code.decode('gbk')
        if char and char.isprintable() and not char.isspace():
            return char
    except (UnicodeDecodeError, LookupError):
        pass
    
    # Try GB18030 (most comprehensive Chinese encoding)
    try:
        char = gb_code.decode('gb18030')
        if char and char.isprintable() and not char.isspace():
            return char
    except (UnicodeDecodeError, LookupError):
        pass
    
    # If all fail, return hex representation for debugging
    hex_repr = gb_code.hex().upper()
    print(f"Warning: Could not decode character with bytes: {hex_repr}")
    return f"[0x{hex_repr}]"

def compute_freeman_code(stroke):
    """Compute Freeman chain code for a stroke."""
    code = []
    for i in range(len(stroke) - 1):
        x1, y1 = stroke[i]
        x2, y2 = stroke[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 360) % 360
        direction = int(round(angle / 45)) % 8
        code.append(DIRECTION_CODES[direction])
    return ''.join(code)

def normalize_character(strokes):
    """Normalize the entire character's strokes to [0,1] range independently for x and y."""
    all_points = [p for stroke in strokes for p in stroke]
    if not all_points:
        return strokes
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1
    normalized_strokes = [
        [((p[0] - min_x) / range_x, (p[1] - min_y) / range_y) for p in stroke]
        for stroke in strokes
    ]
    return normalized_strokes

def setup_chinese_font():
    """Setup Chinese font for matplotlib on different platforms."""
    import matplotlib.font_manager as fm
    import platform
    
    # Try to find and set up Chinese fonts
    system = platform.system()
    chinese_fonts = []
    
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC', 'Songti SC', 'STSong', 'Kaiti SC', 
            'Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti',
            'Apple LiGothic', 'LiHei Pro'
        ]
    elif system == "Windows":
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 
            'Arial Unicode MS', 'DengXian', 'FangSong'
        ]
    else:  # Linux
        chinese_fonts = [
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 
            'AR PL UMing CN', 'DejaVu Sans', 'Droid Sans Fallback'
        ]
    
    # Find available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print("Available Chinese-compatible fonts:")
    chinese_available = [font for font in chinese_fonts if font in available_fonts]
    for font in chinese_available[:5]:  # Show first 5 available
        print(f"  - {font}")
    
    # Set the first available Chinese font
    font_set = False
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"Using font: {font}")
            font_set = True
            break
    
    if not font_set:
        print("Warning: No Chinese font found. Trying system default...")
        # Try some common system fonts that might support Chinese
        fallback_fonts = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
        for font in fallback_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                print(f"Using fallback font: {font}")
                break
    
    # Ensure proper Unicode handling
    plt.rcParams['axes.unicode_minus'] = False
    
    # Test font with a sample Chinese character
    test_chinese_display()

def test_chinese_display():
    """Test if Chinese characters can be displayed properly."""
    try:
        # Create a small test plot
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, '测试', fontsize=20, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Font Test')
        plt.close(fig)  # Close immediately
        print("✓ Chinese character display test passed")
    except Exception as e:
        print(f"⚠ Chinese font test failed: {e}")

def sanitize_filename(char):
    """Convert character to safe filename - KEEP THIS FUNCTION!"""
    if char.startswith('[0x') or char.startswith('U+'):
        # Keep encoded representations as-is but clean them
        return char.replace('[', '').replace(']', '').replace('+', '')
    elif len(char) == 1 and ord(char) > 127:
        # For Chinese characters, use Unicode code point
        return f"U{ord(char):04X}"
    else:
        # For ASCII characters, use as-is but replace problematic chars
        safe_chars = "".join(c if c.isalnum() or c in '-_' else '_' for c in char)
        return safe_chars if safe_chars else f"char_{ord(char[0]):04X}"

def plot_character_interactive(strokes, char, char_index, is_normalized=False, plots_dir=None):
    """Plot the character with color-coded strokes and show interactively."""
    
    # Create new figure
    plt.figure(figsize=(10, 8))
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(strokes)))
    
    # Plot each stroke with different colors
    for i, stroke in enumerate(strokes):
        if len(stroke) > 1:
            x, y = zip(*stroke)
            plt.plot(x, y, color=colors[i], linewidth=2, 
                    marker='o', markersize=3, label=f'Stroke {i+1}')
            
            # Mark start point of each stroke
            plt.plot(x[0], y[0], color=colors[i], marker='s', 
                    markersize=8, markerfacecolor='white', 
                    markeredgecolor=colors[i], markeredgewidth=2)
    
    # Title with character - show the actual Chinese character prominently
    title_type = "Normalized" if is_normalized else "Original"
    
    # Always try to display the character, even if it might be encoded
    title = f"{char} ({title_type}) - Character #{char_index + 1}"
    plt.title(title, fontsize=24, pad=20, fontweight='bold')
    
    # ADD THE YELLOW BOX BACK - This was missing in your second version!
    if not char.startswith('[0x') and not char.startswith('U+'):
        # It's a real character, display it prominently in yellow box
        plt.text(0.02, 0.98, f"字符: {char}", transform=plt.gca().transAxes, 
                fontsize=24, fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    if is_normalized:
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add stroke direction indicators
    plt.xlabel('X coordinate (Square markers = stroke start points)')
    plt.ylabel('Y coordinate')
    
    plt.tight_layout()
    
    # Save plot if directory is provided - USE CHINESE CHARACTER IN FILENAME
    if plots_dir:
        suffix = "normalized" if is_normalized else "original"
        # Use character name for filename with proper sanitization
        safe_char = sanitize_filename(char)
        filename = f"{safe_char}_{suffix}.png"
        filepath = plots_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filepath}")
    
    plt.show()
    
    # Keep the plot window open
    plt.draw()
    plt.pause(0.1)

def save_freeman_codes(strokes, char, char_index, chain_codes_dir):
    """Save Freeman codes to a text file."""
    original_freeman_codes = []
    normalized_strokes = normalize_character(strokes)
    normalized_freeman_codes = []
    
    # Calculate Freeman codes
    for i, stroke in enumerate(strokes):
        original_code = compute_freeman_code(stroke)
        original_freeman_codes.append(original_code)
        
        normalized_code = compute_freeman_code(normalized_strokes[i])
        normalized_freeman_codes.append(normalized_code)
    
    # Create filename using character name - USE CHINESE CHARACTER
    safe_char = sanitize_filename(char)
    filename = f"{safe_char}_freeman_codes.txt"
    filepath = chain_codes_dir / filename
    
    # Write to file with proper encoding
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"CHARACTER: {char}\n")
        
        # Add Unicode information for properly decoded characters
        if not char.startswith('[0x') and len(char) == 1:
            try:
                f.write(f"Unicode: U+{ord(char):04X}\n")
            except:
                pass
                
        f.write("=" * 60 + "\n")
        f.write(f"Number of strokes: {len(strokes)}\n")
        f.write("-" * 60 + "\n\n")
        
        # Write detailed stroke information
        for i, stroke in enumerate(strokes):
            f.write(f"Stroke {i+1}:\n")
            f.write(f"  Points: {len(stroke)} points\n")
            f.write(f"  Start: ({stroke[0][0]}, {stroke[0][1]})\n")
            f.write(f"  End:   ({stroke[-1][0]}, {stroke[-1][1]})\n")
            f.write(f"  Original Freeman Code:   {original_freeman_codes[i]}\n")
            f.write(f"  Normalized Freeman Code: {normalized_freeman_codes[i]}\n")
            
            # Write all points for reference
            f.write(f"  All Points: {stroke}\n")
            f.write("\n")
        
        f.write("FULL FREEMAN CODES:\n")
        f.write(f"Original:   {'|'.join(original_freeman_codes)}\n")
        f.write(f"Normalized: {'|'.join(normalized_freeman_codes)}\n")
        f.write("-" * 60 + "\n")
        
        # Add CSV format for easy processing
        f.write("\nCSV FORMAT:\n")
        f.write("character,stroke_count,original_freeman,normalized_freeman\n")
        f.write(f"\"{char}\",{len(strokes)},\"{'|'.join(original_freeman_codes)}\",\"{'|'.join(normalized_freeman_codes)}\"\n")
    
    print(f"Saved Freeman codes: {filepath}")
    return filepath

def display_freeman_codes(strokes, char, char_index):
    """Display Freeman codes for each stroke in a formatted way."""
    print("=" * 60)
    print(f"CHARACTER: {char}")
    
    # Show Unicode info for properly decoded characters
    if not char.startswith('[0x') and len(char) == 1:
        try:
            print(f"Unicode: U+{ord(char):04X}")
        except:
            pass
            
    print("=" * 60)
    print(f"Number of strokes: {len(strokes)}")
    print("-" * 60)
    
    original_freeman_codes = []
    normalized_strokes = normalize_character(strokes)
    normalized_freeman_codes = []
    
    for i, stroke in enumerate(strokes):
        # Original Freeman code
        original_code = compute_freeman_code(stroke)
        original_freeman_codes.append(original_code)
        
        # Normalized Freeman code
        normalized_code = compute_freeman_code(normalized_strokes[i])
        normalized_freeman_codes.append(normalized_code)
        
        print(f"Stroke {i+1}:")
        print(f"  Points: {len(stroke)} points")
        print(f"  Start: ({stroke[0][0]}, {stroke[0][1]})")
        print(f"  End:   ({stroke[-1][0]}, {stroke[-1][1]})")
        print(f"  Original Freeman Code:   {original_code}")
        print(f"  Normalized Freeman Code: {normalized_code}")
        print()
    
    print("Complete FREEMAN CODES:")
    print(f"Original:   {'|'.join(original_freeman_codes)}")
    print(f"Normalized: {'|'.join(normalized_freeman_codes)}")
    print("-" * 60)

def create_summary_file(characters_processed, chain_codes_dir):
    """Create a summary CSV file with all Freeman codes."""
    summary_file = chain_codes_dir / "freeman_codes_summary.csv"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("character_index,character,character_unicode,stroke_count,original_freeman_codes,normalized_freeman_codes\n")
        
        for i, (char, strokes) in enumerate(characters_processed):
            # Calculate Freeman codes
            original_freeman_codes = []
            normalized_strokes = normalize_character(strokes)
            normalized_freeman_codes = []
            
            for stroke in strokes:
                original_code = compute_freeman_code(stroke)
                original_freeman_codes.append(original_code)
            
            for stroke in normalized_strokes:
                normalized_code = compute_freeman_code(stroke)
                normalized_freeman_codes.append(normalized_code)
            
            # Get Unicode representation
            try:
                if not char.startswith('[0x') and len(char) == 1:
                    unicode_rep = f"U+{ord(char):04X}"
                else:
                    unicode_rep = "ENCODED"
            except:
                unicode_rep = "UNKNOWN"
            
            # Write to CSV
            original_codes_str = '|'.join(original_freeman_codes)
            normalized_codes_str = '|'.join(normalized_freeman_codes)
            
            f.write(f"{i},\"{char}\",{unicode_rep},{len(strokes)},\"{original_codes_str}\",\"{normalized_codes_str}\"\n")
    
    print(f"Created summary file: {summary_file}")

def main():
    """Process the .pot file interactively, showing one character at a time."""
    # Setup Chinese font support first
    print("Setting up Chinese font support...")
    setup_chinese_font()
    
    # Create output directories
    chain_codes_dir = Path("/Users/ballu_macbookpro/Desktop/Thesis/chain_codes")
    plots_dir = Path("/Users/ballu_macbookpro/Desktop/Thesis/plots")
    
    chain_codes_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"Freeman codes: {chain_codes_dir}")
    print(f"Plots: {plots_dir}")

    file_path = Path("/Users/ballu_macbookpro/Desktop/Thesis/C001-f.pot")
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        print("Please update the file_path variable with the correct path to your .pot file")
        return

    print("Reading CASIA POT file...")
    characters = read_pot_file(file_path)
    print(f"Found {len(characters)} characters")
    
    # Show first few characters to verify decoding
    print("\nFirst 5 characters found:")
    for i, (char, strokes) in enumerate(characters[:5]):
        print(f"  {i+1}: {char} ({len(strokes)} strokes)")
    
    print("\nStarting interactive character viewer...")
    print("Press Enter to view each character, type 'quit' to exit, 'skip N' to skip N characters")
    print("-" * 60)

    i = 0
    while i < len(characters):
        char, strokes = characters[i]
        
        # Display Freeman codes in terminal
        display_freeman_codes(strokes, char, i)
        
        # Save Freeman codes to file
        save_freeman_codes(strokes, char, i, chain_codes_dir)
        
        # Plot original strokes
        plot_character_interactive(strokes, char, i, is_normalized=False, plots_dir=plots_dir)
        
        # Plot normalized strokes
        normalized_strokes = normalize_character(strokes)
        plot_character_interactive(normalized_strokes, char, i, is_normalized=True, plots_dir=plots_dir)
        
        # Wait for user input
        user_input = input(f"\nPress Enter for next character, 'b' for previous, 'skip N' to skip N chars, 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit' or user_input == 'q':
            break
        elif user_input == 'b' or user_input == 'back':
            i = max(0, i - 1)
        elif user_input.startswith('skip'):
            try:
                skip_count = int(user_input.split()[1])
                i = min(len(characters) - 1, i + skip_count)
            except (IndexError, ValueError):
                print("Invalid skip command. Use 'skip N' where N is a number.")
                continue
        else:
            i += 1

    print(f"\nSession ended. Processed characters up to #{i+1} of {len(characters)}")
    print(f"Files saved to:")
    print(f"  Freeman codes: {chain_codes_dir}")
    print(f"  Plots: {plots_dir}")
    
    # Create summary file
    create_summary_file(characters[:i+1], chain_codes_dir)

if __name__ == "__main__":
    main()
