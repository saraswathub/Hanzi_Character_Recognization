#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
from typing import Dict, List, Any, Tuple
import time
import urllib.parse
import math
import re

# Define Freeman direction codes (a to h for 0 to 7 directions)
DIRECTION_CODES = 'abcdefgh'

# Direction name mappings
DIRECTION_NAMES = {
    0: '横',  # a=E (horizontal right)
    1: '捺',  # b=NE (diagonal up-right)
    2: '提',  # c=N (vertical up)
    3: '撇',  # d=NW (diagonal up-left)
    4: '横',  # e=W (horizontal left)
    5: '撇',  # f=SW (diagonal down-left)
    6: '竖',  # g=S (vertical down)
    7: '捺'   # h=SE (diagonal down-right)
}

class ChineseStrokeClassifier:
    def __init__(self):
        self.stroke_data = {}
        
        # Eight fundamental stroke types in Chinese calligraphy
        self.stroke_types = {
            'heng': {'name': '横', 'unicode': '㇐', 'description': 'horizontal', 'pattern': 'horizontal_line'},
            'shu': {'name': '竖', 'unicode': '㇑', 'description': 'vertical', 'pattern': 'vertical_line'},
            'pie': {'name': '撇', 'unicode': '㇓', 'description': 'left-falling', 'pattern': 'diagonal_left'},
            'na': {'name': '捺', 'unicode': '㇏', 'description': 'right-falling', 'pattern': 'diagonal_right'},
            'dian': {'name': '点', 'unicode': '㇔', 'description': 'dot', 'pattern': 'dot_or_short'},
            'ti': {'name': '提', 'unicode': '㇀', 'description': 'rising', 'pattern': 'upward_diagonal'},
            'zhe': {'name': '折', 'unicode': '㇕', 'description': 'turning', 'pattern': 'bent_stroke'},
            'gou': {'name': '钩', 'unicode': '㇆', 'description': 'hook', 'pattern': 'hooked_ending'}
        }
        
    def analyze_svg_path(self, path_data: str) -> Dict[str, Any]:
        """Analyze SVG path data to extract geometric properties"""
        if not path_data:
            return {}
            
        points = self._extract_coordinates(path_data)
        if len(points) < 2:
            return {}
            
        start_point = points[0]
        end_point = points[-1]
        
        analysis = {
            'length': self._calculate_path_length(points),
            'start_point': start_point,
            'end_point': end_point,
            'direction_vector': (end_point[0] - start_point[0], end_point[1] - start_point[1]),
            'bounding_box': self._get_bounding_box(points),
            'has_curves': 'Q' in path_data or 'C' in path_data,
            'has_turns': self._detect_turns(points),
            'aspect_ratio': 0,
            'dominant_direction': 'none',
            'points': points,
            'total_points': len(points)
        }
        
        bbox = analysis['bounding_box']
        width = bbox['max_x'] - bbox['min_x']
        height = bbox['max_y'] - bbox['min_y']
        analysis['aspect_ratio'] = width / height if height > 0 else float('inf')
        
        dx = abs(analysis['direction_vector'][0])
        dy = abs(analysis['direction_vector'][1])
        
        if dx > dy * 2:
            analysis['dominant_direction'] = 'horizontal'
        elif dy > dx * 2:
            analysis['dominant_direction'] = 'vertical'
        else:
            analysis['dominant_direction'] = 'diagonal'
            
        return analysis
    
    def generate_freeman_code(self, points: List[Tuple[float, float]], persistence_threshold: int = 3) -> Dict[str, Any]:
        """
        Generate Freeman chain code from stroke points with persistence threshold
        Freeman codes: a=E, b=NE, c=N, d=NW, e=W, f=SW, g=S, h=SE
        """
        if len(points) < 2:
            return {
                'chain_code': '',
                'code_sequence': [],
                'direction_changes': [],
                'direction_names': [],
                'dominant_directions': [],
                'smoothed_code': ''
            }
        
        freeman_codes = []
        min_distance = 5.0
        
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            
            if abs(dx) < min_distance and abs(dy) < min_distance:
                continue
            
            angle = math.atan2(-dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            freeman_index = int((angle + math.pi/8) / (math.pi/4)) % 8
            freeman_codes.append(freeman_index)
        
        letter_codes = [DIRECTION_CODES[code] for code in freeman_codes]
        smoothed_codes = self._smooth_freeman_code(freeman_codes)
        smoothed_letters = [DIRECTION_CODES[code] for code in smoothed_codes]
        direction_names = [DIRECTION_NAMES[code] for code in smoothed_codes]
        
        # Detect significant direction changes with persistence
        direction_changes = []
        if len(direction_names) > 1:
            current_direction = direction_names[0]
            count = 1
            for i in range(1, len(direction_names)):
                if direction_names[i] == current_direction:
                    count += 1
                else:
                    if count >= persistence_threshold:
                        direction_changes.append({
                            'to': current_direction,
                            'position': i - count
                        })
                    current_direction = direction_names[i]
                    count = 1
            if count >= persistence_threshold:
                direction_changes.append({
                    'to': current_direction,
                    'position': len(direction_names) - count
                })
        
        code_counts = {}
        for code in smoothed_codes:
            code_counts[code] = code_counts.get(code, 0) + 1
        
        dominant_directions = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_direction_names = [
            {'direction': DIRECTION_NAMES[code], 'count': count, 'proportion': count / len(smoothed_codes)}
            for code, count in dominant_directions[:3]
        ]
        
        return {
            'chain_code': ''.join(letter_codes),
            'code_sequence': letter_codes,
            'direction_changes': direction_changes,
            'direction_names': direction_names,
            'dominant_directions': dominant_direction_names,
            'smoothed_code': ''.join(smoothed_letters),
            'code_length': len(letter_codes)
        }
    
    def _smooth_freeman_code(self, codes: List[int]) -> List[int]:
        """Apply smoothing to Freeman code sequence"""
        if len(codes) < 3:
            return codes
        
        smoothed = [codes[0]]
        for i in range(1, len(codes)-1):
            window = [codes[i-1], codes[i], codes[i+1]]
            if max(window) - min(window) > 4:
                window = [(x + 4) % 8 for x in window]
                median_val = sorted(window)[1]
                smoothed.append((median_val - 4) % 8)
            else:
                smoothed.append(sorted(window)[1])
        smoothed.append(codes[-1])
        return smoothed

    def classify_stroke_type(self, path_data: str, stroke_index: int = 0) -> Dict[str, Any]:
        """Classify a stroke based on its SVG path data"""
        analysis = self.analyze_svg_path(path_data)
        
        if not analysis:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'chinese_name': '未知',
                'unicode_symbol': '?',
                'reasoning': 'Could not analyze path data',
                'freeman_code': {}
            }
        
        freeman_data = self.generate_freeman_code(analysis['points'])
        classification = self._apply_classification_rules(analysis, stroke_index, freeman_data)
        
        return {
            'type': classification['type'],
            'confidence': classification['confidence'],
            'chinese_name': self.stroke_types.get(classification['type'], {}).get('name', '未知'),
            'unicode_symbol': self.stroke_types.get(classification['type'], {}).get('unicode', '?'),
            'description': self.stroke_types.get(classification['type'], {}).get('description', 'unknown'),
            'reasoning': classification['reasoning'],
            'geometric_analysis': analysis,
            'freeman_code': freeman_data
        }
    
    def _apply_classification_rules(self, analysis: Dict[str, Any], stroke_index: int, freeman_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply heuristic rules to classify stroke type"""
        confidence = 0.0
        stroke_type = 'unknown'
        reasoning = []
        
        length = analysis['length']
        
        if length < 30:
            stroke_type = 'dian'
            confidence = 0.9
            reasoning.append(f"Very short stroke (length: {length:.1f})")
            return {'type': stroke_type, 'confidence': confidence, 'reasoning': '; '.join(reasoning)}
        
        if freeman_data and freeman_data['code_sequence']:
            dominant_dirs = freeman_data['dominant_directions']
            direction_changes = freeman_data['direction_changes']
            code_length = freeman_data['code_length']
            
            if len(direction_changes) <= 2 and code_length > 2:
                if dominant_dirs:
                    main_direction = dominant_dirs[0]['direction']
                    proportion = dominant_dirs[0]['proportion']
                    
                    if proportion > 0.6:
                        if main_direction == '横':
                            stroke_type = 'heng'
                            confidence = 0.95
                            reasoning.append(f"Consistent horizontal direction ({proportion:.1%})")
                        elif main_direction == '竖':
                            stroke_type = 'shu'
                            confidence = 0.95
                            reasoning.append(f"Consistent downward direction ({proportion:.1%})")
                        elif main_direction == '提':
                            stroke_type = 'ti'
                            confidence = 0.9
                            reasoning.append(f"Consistent upward direction ({proportion:.1%})")
                        elif main_direction == '撇':
                            stroke_type = 'pie'
                            confidence = 0.9
                            reasoning.append(f"Consistent left-falling direction ({proportion:.1%})")
                        elif main_direction == '捺':
                            stroke_type = 'na'
                            confidence = 0.9
                            reasoning.append(f"Consistent right-falling direction ({proportion:.1%})")
            
            elif len(direction_changes) >= 3:
                stroke_type = 'zhe'
                confidence = 0.8
                reasoning.append(f"Multiple significant direction changes: {len(direction_changes)} detected")
        
        if confidence < 0.8:
            dx, dy = analysis['direction_vector']
            aspect_ratio = analysis['aspect_ratio']
            
            if aspect_ratio > 3:
                stroke_type = 'heng'
                confidence = 0.85
                reasoning.append(f"High aspect ratio ({aspect_ratio:.1f}) suggests horizontal")
            elif aspect_ratio < 0.33:
                if dy > 0:
                    stroke_type = 'shu'
                    confidence = 0.85
                    reasoning.append("Tall aspect ratio with downward direction")
                else:
                    stroke_type = 'ti'
                    confidence = 0.8
                    reasoning.append("Tall aspect ratio with upward direction")
            else:
                if abs(dx) > abs(dy):
                    if dx > 0:
                        stroke_type = 'na' if dy > 0 else 'ti'
                        confidence = 0.75
                        reasoning.append("Diagonal stroke, right-leaning")
                    else:
                        stroke_type = 'pie'
                        confidence = 0.75
                        reasoning.append("Diagonal stroke, left-leaning")
                else:
                    if dy > 0:
                        stroke_type = 'shu'
                        confidence = 0.8
                        reasoning.append("Primarily vertical downward")
                    else:
                        stroke_type = 'ti'
                        confidence = 0.75
                        reasoning.append("Primarily vertical upward")
        
        if self._has_hook_ending(analysis):
            if stroke_type in ['shu', 'heng']:
                stroke_type = 'gou'
                confidence = 0.85
                reasoning.append("Detected hook ending")
        
        if stroke_index == 0 and stroke_type == 'heng':
            confidence = min(confidence + 0.05, 1.0)
            reasoning.append("First stroke boost")
        
        return {
            'type': stroke_type,
            'confidence': min(confidence, 1.0),
            'reasoning': '; '.join(reasoning)
        }
    
    def _extract_coordinates(self, path_data: str) -> List[Tuple[float, float]]:
        """Extract coordinate points from SVG path data"""
        points = []
        numbers = re.findall(r'-?\d+\.?\d*', path_data)
        
        for i in range(0, len(numbers)-1, 2):
            try:
                x = float(numbers[i])
                y = float(numbers[i+1])
                points.append((x, y))
            except (ValueError, IndexError):
                continue
                
        return points
    
    def _calculate_path_length(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total path length"""
        if len(points) < 2:
            return 0
            
        total_length = 0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
            
        return total_length
    
    def _get_bounding_box(self, points: List[Tuple[float, float]]) -> Dict[str, float]:
        """Get bounding box of points"""
        if not points:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
            
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        return {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys)
        }
    
    def _detect_turns(self, points: List[Tuple[float, float]]) -> bool:
        """Detect if path has significant turns"""
        if len(points) < 4:
            return False
            
        directions = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            if abs(dx) > 3 or abs(dy) > 3:
                angle = math.atan2(dy, dx)
                directions.append(angle)
        
        if len(directions) < 3:
            return False
        
        significant_turns = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.pi/4:
                significant_turns += 1
                
        return significant_turns >= 1
    
    def _has_hook_ending(self, analysis: Dict[str, Any]) -> bool:
        """Detect if stroke ends with a hook"""
        points = analysis.get('points', [])
        if len(points) < 4:
            return False
            
        last_points = points[-4:]
        if len(last_points) < 4:
            return False
            
        directions = []
        for i in range(1, len(last_points)):
            dx = last_points[i][0] - last_points[i-1][0]
            dy = last_points[i][1] - last_points[i-1][1]
            if abs(dx) > 1 or abs(dy) > 1:
                directions.append(math.atan2(dy, dx))
        
        if len(directions) < 2:
            return False
            
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.pi/2:
                return True
                
        return False
    
    def get_stroke_order_with_classification(self, char: str) -> Dict[str, Any]:
        """Get stroke order data and classify each stroke type"""
        try:
            encoded_char = urllib.parse.quote(char, safe='')
            urls = [
                f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{encoded_char}.json",
                f"https://raw.githubusercontent.com/chanind/hanzi-writer-data/master/{encoded_char}.json",
                f"https://unpkg.com/hanzi-writer-data/{encoded_char}.json"
            ]
            
            for i, url in enumerate(urls, 1):
                print(f"  [{i}/{len(urls)}] Trying: {url}")
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  ✓ SUCCESS: Found stroke data!")
                        classified_strokes = []
                        strokes = data.get('strokes', [])
                        print(f"  🔍 Classifying {len(strokes)} strokes...")
                        
                        for idx, stroke_path in enumerate(strokes):
                            classification = self.classify_stroke_type(stroke_path, idx)
                            classified_strokes.append({
                                'stroke_index': idx + 1,
                                'svg_path': stroke_path,
                                'classification': classification,
                                'stroke_order_position': idx + 1
                            })
                            print(f"    Stroke {idx+1}: {classification['chinese_name']} ({classification['type']}) - {classification['confidence']:.2f}")
                            if classification['freeman_code']['direction_names']:
                                directions = " → ".join(classification['freeman_code']['direction_names'])
                                print(f"      Directions: {directions}")
                            if classification['freeman_code']['direction_changes']:
                                changes = [f"{c['to']}" for c in classification['freeman_code']['direction_changes']]
                                print(f"      Direction changes: {', '.join(changes)}")
                        
                        return {
                            'character': char,
                            'total_strokes': len(strokes),
                            'classified_strokes': classified_strokes,
                            'medians': data.get('medians', []),
                            'api_url': url,
                            'stroke_sequence': [s['classification']['chinese_name'] for s in classified_strokes],
                            'stroke_types_summary': self._get_stroke_summary(classified_strokes),
                            'raw_data': data
                        }
                except requests.exceptions.RequestException as e:
                    print(f"    Network error: {str(e)[:100]}...")
                    continue
            return {}
        except Exception as e:
            print(f"    Unexpected error: {e}")
            return {}
    
    def _get_stroke_summary(self, classified_strokes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics of stroke types"""
        type_counts = {}
        confidence_scores = []
        
        for stroke in classified_strokes:
            stroke_type = stroke['classification']['type']
            confidence = stroke['classification']['confidence']
            type_counts[stroke_type] = type_counts.get(stroke_type, 0) + 1
            confidence_scores.append(confidence)
        
        return {
            'stroke_type_distribution': type_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'low_confidence_strokes': sum(1 for s in confidence_scores if s < 0.7),
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else 'none'
        }
    
    def process_characters_with_classification(self, characters: List[str]) -> Dict[str, Any]:
        """Process characters and return classified stroke data"""
        print(f"\n Processing {len(characters)} characters with stroke classification...")
        print("=" * 80)
        results = {}
        
        for i, char in enumerate(characters, 1):
            print(f"\n[{i}/{len(characters)}] Processing: {char}")
            print("-" * 40)
            classified_data = self.get_stroke_order_with_classification(char)
            if classified_data:
                results[char] = classified_data
                summary = classified_data['stroke_types_summary']
                print(f"  ✓ Total strokes: {classified_data['total_strokes']}")
                print(f"  ✓ Stroke sequence: {' → '.join(classified_data['stroke_sequence'])}")
                print(f"  ✓ Most common type: {summary['most_common_type']}")
                print(f"  ✓ Average confidence: {summary['average_confidence']:.2f}")
                if summary['low_confidence_strokes'] > 0:
                    print(f"  ⚠️  {summary['low_confidence_strokes']} low-confidence classifications")
            else:
                print(f"  ✗ Failed to process {char}")
                results[char] = {'character': char, 'error': 'Could not retrieve or classify strokes'}
            time.sleep(1)
        return results
    
    def export_training_data(self, classified_data: Dict[str, Any], filename: str = 'chinese_stroke_training_data.json'):
        """Export data in format suitable for ML training"""
        training_data = {
            'metadata': {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_characters': len(classified_data),
                'stroke_types': list(self.stroke_types.keys()),
                'stroke_type_definitions': self.stroke_types,
                'classification_method': 'geometric_heuristics',
                'coordinate_system': '1024x1024_hanzi_writer',
                'freeman_code_system': 'letter_based_a_to_h',
                'direction_names': ['横', '竖', '撇', '捺', '提']
            },
            'characters': {},
            'training_examples': []
        }
        
        for char, data in classified_data.items():
            if 'classified_strokes' not in data:
                continue
            training_data['characters'][char] = {
                'unicode': ord(char),
                'total_strokes': data['total_strokes'],
                'stroke_sequence': data['stroke_sequence'],
                'strokes': []
            }
            for stroke_info in data['classified_strokes']:
                stroke_example = {
                    'character': char,
                    'stroke_index': stroke_info['stroke_index'],
                    'stroke_type': stroke_info['classification']['type'],
                    'chinese_name': stroke_info['classification']['chinese_name'],
                    'confidence': stroke_info['classification']['confidence'],
                    'svg_path': stroke_info['svg_path'],
                    'geometric_features': stroke_info['classification']['geometric_analysis'],
                    'classification_reasoning': stroke_info['classification']['reasoning'],
                    'freeman_code': stroke_info['classification']['freeman_code']
                }
                training_data['characters'][char]['strokes'].append(stroke_example)
                training_data['training_examples'].append(stroke_example)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n Training data exported to {filename}")
        print(f" Total training examples: {len(training_data['training_examples'])}")
        print(f" Characters processed: {len(training_data['characters'])}")
        type_counts = {}
        for example in training_data['training_examples']:
            stroke_type = example['stroke_type']
            type_counts[stroke_type] = type_counts.get(stroke_type, 0) + 1
        print(f"\n Stroke Type Distribution:")
        for stroke_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            chinese_name = self.stroke_types.get(stroke_type, {}).get('name', stroke_type)
            percentage = count / len(training_data['training_examples']) * 100
            print(f"   {chinese_name} ({stroke_type}): {count} ({percentage:.1f}%)")

def get_user_characters() -> List[str]:
    """Get Chinese characters from user input"""
    print("\n" + "="*60)
    print(" CHARACTER INPUT")
    print("="*60)
    user_input = input("\n  Enter characters: ").strip()
    if not user_input:
        print(" No characters entered.")
        return []
    characters = []
    tokens = [token.strip() for token in user_input.split(',') if token.strip()]
    for token in tokens:
        for char in token:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                if char not in characters:
                    characters.append(char)
            elif char.strip():
                print(f"⚠️  Skipping non-Chinese character: '{char}'")
    if not characters:
        print(" No valid Chinese characters found.")
        return []
    print(f" Found {len(characters)} Chinese characters: {','.join(characters)}")
    return characters

def main():
    print("Enhanced Chinese Character Stroke Classifier with Direction Names")
    print("Extracts SVG paths, Freeman chain codes (a-h), AND classifies stroke types")
    print("Shows actual direction names: 横, 竖, 撇, 捺, 提")
    print("=" * 80)
    classifier = ChineseStrokeClassifier()
    characters = get_user_characters()
    if not characters:
        print("No characters to process. Exiting.")
        return
    print(f"\nProcessing {len(characters)} characters: {','.join(characters)}")
    char_sample = ''.join(characters[:5])
    if len(characters) > 5:
        char_sample += f"_and_{len(characters)-5}_more"
    output_filename = f"stroke_analysis_{char_sample}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    print(f" Will save results to: {output_filename}")
    results = classifier.process_characters_with_classification(characters)
    classifier.export_training_data(results, output_filename)
    print(f"\n Analysis complete!")

if __name__ == "__main__":
    try:
        import requests
        import urllib.parse
    except ImportError as e:
        print(f" Error: Required library missing: {e}")
        print(" Install with: pip install requests")
        exit(1)
    main()
