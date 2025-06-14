import struct
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Define Freeman direction codes (a to h for 0 to 7 directions)
DIRECTION_CODES = 'abcdefgh'

def read_pot_file(file_path):
    """Read the .pot file and extract character data and strokes with improved error handling."""
    characters = []
    char_count = 0
    
    try:
        with open(file_path, 'rb') as f:
            print(f"Reading POT file: {file_path}")
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size:,} bytes")
            
            while True:
                current_pos = f.tell()
                if current_pos >= file_size - 8:  # Not enough bytes for a complete record
                    break
                
                # Read sample size (2 bytes)
                sample_size_data = f.read(2)
                if len(sample_size_data) < 2:
                    break
                sample_size = struct.unpack('<H', sample_size_data)[0]
                
                # Check for end of file marker
                if sample_size == 0:
                    print(f"Found end marker at position {current_pos}")
                    break

                # Read tag code (4 bytes, but only first 2 are meaningful)
                tag_code_data = f.read(4)
                if len(tag_code_data) < 4:
                    print(f"Warning: Incomplete tag code data at position {current_pos}")
                    break
                gb_code = tag_code_data[:2]

                # Improved Chinese character decoding
                char = decode_chinese_character(gb_code)

                # Read stroke number (2 bytes)
                stroke_num_data = f.read(2)
                if len(stroke_num_data) < 2:
                    print(f"Warning: Incomplete stroke number data at position {current_pos}")
                    break
                stroke_num = struct.unpack('<H', stroke_num_data)[0]

                # Validate stroke number
                if stroke_num > 100:  # Reasonable upper bound for stroke count
                    print(f"Warning: Unusually high stroke count {stroke_num} for character {char}")
                    continue

                # Read strokes
                strokes = []
                stroke_points_read = 0
                
                for stroke_idx in range(stroke_num):
                    stroke = []
                    points_in_stroke = 0
                    
                    while True:
                        point_data = f.read(4)
                        if len(point_data) < 4:
                            print(f"Warning: Incomplete point data in stroke {stroke_idx+1}")
                            break
                        
                        x, y = struct.unpack('<hh', point_data)
                        stroke_points_read += 1
                        
                        # Check for stroke end marker
                        if x == -1 and y == 0:
                            break
                        
                        stroke.append((x, y))
                        points_in_stroke += 1
                        
                        # Safety check to prevent infinite loops
                        if points_in_stroke > 1000:
                            print(f"Warning: Too many points in stroke {stroke_idx+1}, possible data corruption")
                            break
                    
                    if len(stroke) > 0:  # Only add non-empty strokes
                        strokes.append(stroke)

                # Read character end marker (4 bytes)
                end_marker_data = f.read(4)
                if len(end_marker_data) < 4:
                    print(f"Warning: Missing end marker for character {char}")
                    break
                
                end_x, end_y = struct.unpack('<hh', end_marker_data)
                if end_x != -1 or end_y != -1:
                    print(f"Warning: Invalid end marker ({end_x}, {end_y}) for character {char}")
                    # Continue processing but note the issue

                # Only add characters with valid strokes
                if strokes and len(strokes) > 0:
                    characters.append((char, strokes))
                    char_count += 1
                    
                    # Progress indicator
                    if char_count % 100 == 0:
                        print(f"Processed {char_count} characters...")

    except Exception as e:
        print(f"Error reading POT file: {e}")
        print(f"Successfully read {char_count} characters before error")
    
    print(f"Total characters read: {len(characters)}")
    return characters

def decode_chinese_character(gb_code):
    """Improved Chinese character decoding with better error handling."""
    # Skip null bytes
    if gb_code == b'\x00\x00':
        return "[NULL]"
    
    # Try different Chinese encodings in order of likelihood
    encodings = ['gb2312', 'gbk', 'gb18030']
    
    for encoding in encodings:
        try:
            char = gb_code.decode(encoding)
            # Validate that it's a printable, non-whitespace character
            if char and char.isprintable() and not char.isspace() and ord(char) > 127:
                return char
        except (UnicodeDecodeError, LookupError):
            continue
    
    # If all encodings fail, return hex representation
    hex_repr = gb_code.hex().upper()
    return f"[0x{hex_repr}]"

def compute_freeman_code(stroke):
    """Compute Freeman chain code for a stroke with improved direction calculation."""
    if len(stroke) < 2:
        return ""
    
    code = []
    for i in range(len(stroke) - 1):
        x1, y1 = stroke[i]
        x2, y2 = stroke[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        
        # Skip points that are the same
        if dx == 0 and dy == 0:
            continue
        
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize to 0-360 range
        angle = (angle + 360) % 360
        
        # Convert to Freeman direction (0-7)
        # Freeman codes: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        direction = int(round(angle / 45)) % 8
        code.append(DIRECTION_CODES[direction])
    
    return ''.join(code)

def normalize_character(strokes):
    """Normalize character strokes to [0,1] range with aspect ratio preservation option."""
    if not strokes:
        return strokes
    
    # Get all points
    all_points = [p for stroke in strokes for p in stroke]
    if not all_points:
        return strokes
    
    # Find bounding box
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    # Calculate ranges
    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1
    
    # Normalize each stroke
    normalized_strokes = []
    for stroke in strokes:
        normalized_stroke = [
            ((p[0] - min_x) / range_x, (p[1] - min_y) / range_y) 
            for p in stroke
        ]
        normalized_strokes.append(normalized_stroke)
    
    return normalized_strokes

def setup_chinese_font():
    """Setup Chinese font for matplotlib with comprehensive platform support."""
    import matplotlib.font_manager as fm
    import platform
    
    print("Setting up Chinese font support...")
    
    # Get system info
    system = platform.system()
    print(f"Operating system: {system}")
    
    # Define font preferences by platform
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC', 'Songti SC', 'STSong', 'Kaiti SC', 
            'Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti',
            'Apple LiGothic', 'LiHei Pro', 'Heiti SC'
        ]
    elif system == "Windows":
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 
            'Microsoft JhengHei', 'Arial Unicode MS', 'DengXian', 
            'FangSong', 'NSimSun'
        ]
    else:  # Linux and others
        chinese_fonts = [
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 
            'WenQuanYi Zen Hei', 'AR PL UMing CN', 
            'DejaVu Sans', 'Droid Sans Fallback'
        ]
    
    # Get available fonts
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    print(f"Total available fonts: {len(available_fonts)}")
    
    # Find Chinese-compatible fonts
    chinese_available = [font for font in chinese_fonts if font in available_fonts]
    print(f"Available Chinese fonts: {chinese_available[:5]}")  # Show first 5
    
    # Set font
    font_set = False
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"✓ Using font: {font}")
            font_set = True
            break
    
    if not font_set:
        print("⚠ No preferred Chinese font found, using system default")
        # Try fallback fonts
        fallback_fonts = ['Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
        for font in fallback_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                print(f"Using fallback font: {font}")
                break
    
    # Configure matplotlib for Chinese text
    plt.rcParams['axes.unicode_minus'] = False
    
    # Test Chinese display
    test_chinese_display()

def test_chinese_display():
    """Test Chinese character display capability."""
    try:
        fig, ax = plt.subplots(figsize=(2, 1))
        test_chars = ['中', '文', '字']
        ax.text(0.5, 0.5, ''.join(test_chars), fontsize=16, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.close(fig)
        print("✓ Chinese character display test passed")
        return True
    except Exception as e:
        print(f"⚠ Chinese font test failed: {e}")
        return False

def sanitize_filename(char):
    """Convert character to filesystem-safe filename."""
    # Handle encoded representations
    if char.startswith('[') and char.endswith(']'):
        return char[1:-1].replace('0x', 'hex_')
    
    # Handle Unicode characters
    if len(char) == 1 and ord(char) > 127:
        return f"U{ord(char):04X}"
    
    # Handle ASCII characters
    safe_chars = ''.join(c if c.isalnum() or c in '-_' else '_' for c in char)
    return safe_chars if safe_chars else f"char_{ord(char[0]):04X}"

def plot_character_with_analysis(strokes, char, char_index, output_dir=None):
    """Create comprehensive character visualization with analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # Normalize strokes
    normalized_strokes = normalize_character(strokes)
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(strokes)))
    
    # Check if character can be displayed (not encoded)
    is_displayable = not (char.startswith('[') or char.startswith('U+') or len(char) != 1)
    
    # Plot 1: Original strokes with character overlay
    ax1.set_title(f'Original Strokes - {char}', fontsize=16, fontweight='bold', pad=20)
    for i, stroke in enumerate(strokes):
        if len(stroke) > 1:
            x, y = zip(*stroke)
            ax1.plot(x, y, color=colors[i], linewidth=2.5, marker='o', markersize=3)
            ax1.plot(x[0], y[0], 's', color=colors[i], markersize=10, markerfacecolor='white', markeredgewidth=2)
            ax1.text(x[0], y[0], str(i+1), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add large Chinese character overlay in corner
    if is_displayable:
        ax1.text(0.95, 0.05, char, transform=ax1.transAxes, fontsize=80, 
                ha='right', va='bottom', alpha=0.3, color='red', fontweight='bold')
        # Also add it in a prominent yellow box
        ax1.text(0.02, 0.98, f'字: {char}', transform=ax1.transAxes, fontsize=24, 
                ha='left', va='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X coordinate', fontsize=12)
    ax1.set_ylabel('Y coordinate', fontsize=12)
    
    # Plot 2: Normalized strokes with character overlay
    ax2.set_title(f'Normalized Strokes - {char}', fontsize=16, fontweight='bold', pad=20)
    for i, stroke in enumerate(normalized_strokes):
        if len(stroke) > 1:
            x, y = zip(*stroke)
            ax2.plot(x, y, color=colors[i], linewidth=2.5, marker='o', markersize=3)
            ax2.plot(x[0], y[0], 's', color=colors[i], markersize=10, markerfacecolor='white', markeredgewidth=2)
            ax2.text(x[0], y[0], str(i+1), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add large Chinese character overlay in corner
    if is_displayable:
        ax2.text(0.95, 0.05, char, transform=ax2.transAxes, fontsize=80, 
                ha='right', va='bottom', alpha=0.3, color='red', fontweight='bold')
        # Also add it in a prominent yellow box
        ax2.text(0.02, 0.98, f'字: {char}', transform=ax2.transAxes, fontsize=24, 
                ha='left', va='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Normalized X', fontsize=12)
    ax2.set_ylabel('Normalized Y', fontsize=12)
    
    # Plot 3: Character display and Freeman codes
    ax3.set_title('Character & Freeman Codes', fontsize=16, fontweight='bold', pad=20)
    
    # Large character display at top
    if is_displayable:
        ax3.text(0.5, 0.85, char, transform=ax3.transAxes, fontsize=120, 
                ha='center', va='center', fontweight='bold', color='darkblue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    else:
        ax3.text(0.5, 0.85, char, transform=ax3.transAxes, fontsize=24, 
                ha='center', va='center', fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # Freeman codes below
    freeman_codes = []
    y_pos = 0.65
    for i, stroke in enumerate(strokes):
        code = compute_freeman_code(stroke)
        freeman_codes.append(code)
        if y_pos > 0.1:  # Only show if there's space
            ax3.text(0.05, y_pos, f'Stroke {i+1}: {code}', 
                    transform=ax3.transAxes, fontsize=11, fontfamily='monospace')
            y_pos -= 0.08
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Plot 4: Character statistics with large character
    ax4.set_title('Statistics & Info', fontsize=16, fontweight='bold', pad=20)
    
    # Character display at top right
    if is_displayable:
        ax4.text(0.85, 0.85, char, transform=ax4.transAxes, fontsize=60, 
                ha='center', va='center', alpha=0.4, color='green', fontweight='bold')
    
    # Statistics
    unicode_val = f"U+{ord(char):04X}" if len(char) == 1 and ord(char) > 127 else "N/A"
    stats_text = f"""CHARACTER: {char}

Index: {char_index + 1}
Unicode: {unicode_val}
Stroke Count: {len(strokes)}
Total Points: {sum(len(stroke) for stroke in strokes)}

FREEMAN CODES:
{chr(10).join(f"S{i+1}: {compute_freeman_code(stroke)}" for i, stroke in enumerate(strokes[:8]))}
{"..." if len(strokes) > 8 else ""}
    """
    ax4.text(0.05, 0.95, stats_text.strip(), transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Add overall figure title with the character
    if is_displayable:
        fig.suptitle(f'Character Analysis: {char} (#{char_index + 1})', 
                    fontsize=20, fontweight='bold', y=0.95)
    else:
        fig.suptitle(f'Character Analysis: {char} (#{char_index + 1})', 
                    fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save plot if output directory specified
    if output_dir:
        safe_char = sanitize_filename(char)
        filename = f"{char_index+1:04d}_{safe_char}_analysis.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved analysis plot: {filepath}")
    
    plt.show()
    return freeman_codes

def create_character_comparison_plot(strokes, char, char_index, output_dir=None):
    """Create a side-by-side comparison: actual character vs stroke pattern."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Check if character can be displayed
    is_displayable = not (char.startswith('[') or char.startswith('U+') or len(char) != 1)
    
    # Plot 1: Large character display
    ax1.set_title('Actual Character', fontsize=18, fontweight='bold', pad=20)
    if is_displayable:
        ax1.text(0.5, 0.5, char, transform=ax1.transAxes, fontsize=200, 
                ha='center', va='center', fontweight='bold', color='darkblue')
        
        # Add character info
        unicode_val = f"U+{ord(char):04X}" if ord(char) > 127 else "ASCII"
        ax1.text(0.5, 0.1, f'{unicode_val}', transform=ax1.transAxes, fontsize=16, 
                ha='center', va='center', style='italic')
    else:
        ax1.text(0.5, 0.5, char, transform=ax1.transAxes, fontsize=48, 
                ha='center', va='center', fontweight='bold', color='darkred')
        ax1.text(0.5, 0.3, 'Encoded Character', transform=ax1.transAxes, fontsize=14, 
                ha='center', va='center', style='italic', color='gray')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_facecolor('lightyellow')
    
    # Plot 2: Stroke pattern (original)
    ax2.set_title('Stroke Pattern (Original)', fontsize=18, fontweight='bold', pad=20)
    colors = plt.cm.Set1(np.linspace(0, 1, len(strokes)))
    
    for i, stroke in enumerate(strokes):
        if len(stroke) > 1:
            x, y = zip(*stroke)
            ax2.plot(x, y, color=colors[i], linewidth=3, marker='o', markersize=4, 
                    label=f'Stroke {i+1}', alpha=0.8)
            # Mark stroke start
            ax2.plot(x[0], y[0], 's', color=colors[i], markersize=12, 
                    markerfacecolor='white', markeredgewidth=3)
            # Number the stroke
            ax2.text(x[0], y[0], str(i+1), ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlabel('X coordinate', fontsize=12)
    ax2.set_ylabel('Y coordinate', fontsize=12)
    
    # Plot 3: Stroke pattern (normalized) with character overlay
    ax3.set_title('Normalized Pattern + Character', fontsize=18, fontweight='bold', pad=20)
    normalized_strokes = normalize_character(strokes)
    
    for i, stroke in enumerate(normalized_strokes):
        if len(stroke) > 1:
            x, y = zip(*stroke)
            ax3.plot(x, y, color=colors[i], linewidth=3, marker='o', markersize=4, alpha=0.8)
            ax3.plot(x[0], y[0], 's', color=colors[i], markersize=12, 
                    markerfacecolor='white', markeredgewidth=3)
            ax3.text(x[0], y[0], str(i+1), ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    # Overlay the character transparently
    if is_displayable:
        ax3.text(0.5, 0.5, char, transform=ax3.transAxes, fontsize=120, 
                ha='center', va='center', alpha=0.2, color='red', fontweight='bold')
    
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Normalized X', fontsize=12)
    ax3.set_ylabel('Normalized Y', fontsize=12)
    
    # Add overall title
    fig.suptitle(f'Character Analysis: {char} (#{char_index + 1}) - {len(strokes)} strokes', 
                fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save if directory provided
    if output_dir:
        safe_char = sanitize_filename(char)
        filename = f"{char_index+1:04d}_{safe_char}_comparison.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {filepath}")
    
    plt.show()

def save_character_data(char, strokes, char_index, output_dir):
    """Save comprehensive character data to files."""
    safe_char = sanitize_filename(char)
    
    # Freeman codes
    original_codes = [compute_freeman_code(stroke) for stroke in strokes]
    normalized_strokes = normalize_character(strokes)
    normalized_codes = [compute_freeman_code(stroke) for stroke in normalized_strokes]
    
    # Save detailed data
    data_file = output_dir / f"{char_index+1:04d}_{safe_char}_data.txt"
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(f"Character: {char}\n")
        f.write(f"Index: {char_index + 1}\n")
        f.write(f"Unicode: {f'U+{ord(char):04X}' if len(char) == 1 and ord(char) > 127 else 'N/A'}\n")
        f.write(f"Stroke Count: {len(strokes)}\n")
        f.write(f"Total Points: {sum(len(stroke) for stroke in strokes)}\n")
        f.write("-" * 50 + "\n")
        
        for i, stroke in enumerate(strokes):
            f.write(f"\nStroke {i+1}:\n")
            f.write(f"  Points: {len(stroke)}\n")
            f.write(f"  Coordinates: {stroke}\n")
            f.write(f"  Freeman Code: {original_codes[i]}\n")
            f.write(f"  Normalized Freeman Code: {normalized_codes[i]}\n")
    
    return data_file

def main():
    """Main interactive processing function."""
    # Setup
    print("Chinese Character POT File Processor")
    print("=" * 50)
    
    setup_chinese_font()
    
    # Define file paths - UPDATE THESE PATHS
    pot_file = Path("/Users/ballu_macbookpro/Desktop/Thesis/C001-f.pot")
    output_base = Path("/Users/ballu_macbookpro/Desktop/Thesis/output")
    
    # Create output directories
    plots_dir = output_base / "plots"
    data_dir = output_base / "data"
    
    for dir_path in [plots_dir, data_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Check file existence
    if not pot_file.exists():
        print(f"❌ Error: POT file not found: {pot_file}")
        print("Please update the pot_file path in the script.")
        return
    
    # Read characters
    print(f"\nReading POT file: {pot_file}")
    characters = read_pot_file(pot_file)
    
    if not characters:
        print("❌ No characters found in the POT file.")
        return
    
    print(f"✓ Successfully loaded {len(characters)} characters")
    
    # Show sample characters
    print("\nFirst 10 characters:")
    for i, (char, strokes) in enumerate(characters[:10]):
        print(f"  {i+1:2d}: {char} ({len(strokes)} strokes)")
    
    # Interactive processing
    print("\n" + "=" * 50)
    print("Interactive Character Processor")
    print("Commands: Enter=next, 'b'=back, 'skip N'=skip N chars, 'quit'=exit")
    print("=" * 50)
    
    i = 0
    processed_chars = []
    
    try:
        while i < len(characters):
            char, strokes = characters[i]
            
            print(f"\nProcessing character {i+1}/{len(characters)}: {char}")
            
            # Show character comparison plot first (clearer view)
            create_character_comparison_plot(strokes, char, i, plots_dir)
            
            # Then show detailed analysis
            freeman_codes = plot_character_with_analysis(strokes, char, i, plots_dir)
            
            # Save data
            data_file = save_character_data(char, strokes, i, data_dir)
            processed_chars.append((char, strokes, freeman_codes))
            
            # User interaction
            user_input = input(f"\nContinue? (Enter/b/skip N/quit): ").strip().lower()
            
            if user_input in ['quit', 'q', 'exit']:
                break
            elif user_input in ['b', 'back']:
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
    
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
    
    # Generate summary
    print(f"\n✓ Session completed. Processed {len(processed_chars)} characters.")
    print(f"📁 Output saved to: {output_base}")
    
    # Create summary CSV
    summary_file = output_base / "processing_summary.csv"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("index,character,unicode,stroke_count,freeman_codes\n")
        for idx, (char, strokes, codes) in enumerate(processed_chars):
            unicode_val = f"U+{ord(char):04X}" if len(char) == 1 and ord(char) > 127 else "N/A"
            codes_str = "|".join(codes)
            f.write(f"{idx},{char},{unicode_val},{len(strokes)},\"{codes_str}\"\n")
    
    print(f"📊 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()