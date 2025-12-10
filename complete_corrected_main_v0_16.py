import numpy as np
import pandas as pd
import math, os, random
import re
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, GlobalAveragePooling2D, Flatten, \
    AveragePooling2D, Dense, LSTM
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import json

print("=" * 60)
print("SCRIPT INITIALIZATION STARTED")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ===== Configuration =====
data_folder = "/Users/ballu_macbookpro/Downloads/Hanzi_Recognition_Project/Training"
save_results_root = os.path.expanduser("~/Downloads/results_stknet_hanzi_freeman_weighted")
os.makedirs(save_results_root, exist_ok=True)

print(f"\nChecking data folder: {data_folder}")
print(f"Folder exists: {os.path.exists(data_folder)}")
if not os.path.exists(data_folder):
    print("ERROR: Data folder not found! Please check the path.")
    print("Exiting...")
    exit(1)

# Find all Excel files in the folder
excel_files = []
for file in os.listdir(data_folder):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        excel_files.append(os.path.join(data_folder, file))

print(f"\nFound {len(excel_files)} Excel files:")
for f in excel_files:
    print(f"  - {os.path.basename(f)}")

if len(excel_files) == 0:
    print("ERROR: No Excel files found in the folder!")
    print("Exiting...")
    exit(1)

desired_length = 25
margin = 0.3
noise_level_small = 1 / 3
random_seed = 1000

np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# CJK font support
try:
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# CONFIGURATION - Adjusted for your data
limit_classes_to_top = 50  # Reduced from 200
min_samples_per_selected_class = 2  # Reduced from 3 to be more lenient

# ===== Direction-Aware Freeman Code Distance =====
def freeman_angular_difference(code1: int, code2: int) -> float:
    """Compute angular difference between two Freeman codes (0-7)."""
    diff = abs(code1 - code2)
    if diff > 4:
        diff = 8 - diff
    return diff / 4.0


def weighted_freeman_edit_distance(seq1: str, seq2: str) -> float:
    """
    FIXED: Compute weighted edit distance between Freeman code sequences.
    Now uses consistent parsing with parse_freeman_string().
    """
    # FIX: Use same parsing method as everywhere else
    parsed1 = re.findall(r'[0-7]', str(seq1))
    parsed2 = re.findall(r'[0-7]', str(seq2))
    
    # Join and filter valid codes
    s1 = ''.join(c for c in parsed1 if c.isdigit() and len(c) == 1 and int(c) <= 7)
    s2 = ''.join(c for c in parsed2 if c.isdigit() and len(c) == 1 and int(c) <= 7)
    
    if not s1 and not s2:
        return 0.0
    
    n, m = len(s1), len(s2)
    
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            code1 = int(s1[i-1])
            code2 = int(s2[j-1])
            
            if code1 == code2:
                substitution_cost = 0.0
            else:
                substitution_cost = freeman_angular_difference(code1, code2)
            
            dp[i][j] = min(
                dp[i-1][j] + 1.0,
                dp[i][j-1] + 1.0,
                dp[i-1][j-1] + substitution_cost
            )
    
    max_len = max(n, m)
    if max_len == 0:
        return 0.0
    
    return dp[n][m] / max_len


# ===== Freeman Chain Code Utilities =====
FREEMAN_8 = {
    0: (1, 0),   1: (1, 1),   2: (0, 1),   3: (-1, 1),
    4: (-1, 0),  5: (-1,-1),  6: (0,-1),  7: (1,-1)
}

def parse_freeman_string(s: str) -> List[str]:
    """Parse Freeman code string, extracting digit sequences."""
    return re.findall(r'[0-7]', str(s))

def freeman_to_xy(codes: List[int], start=(0.0, 0.0)) -> np.ndarray:
    """Convert Freeman codes to cumulative 2D coordinates with pen info."""
    x, y = start
    pts = [(x, y)]
    for c in codes:
        dx, dy = FREEMAN_8.get(c, (0, 0))
        x += dx; y += dy
        pts.append((x, y))
    xy = np.array(pts, dtype=np.float32)
    pen = np.ones((xy.shape[0], 1), dtype=np.float32)
    return np.concatenate([xy, pen], axis=1)

def normalize_xy(traj: np.ndarray, eps=1e-6) -> np.ndarray:
    """Zero-center and scale XY to unit square."""
    if traj.size == 0:
        return traj
    xy = traj[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    size = np.maximum(max_xy - min_xy, eps)
    scale = np.max(size)
    if scale < eps:
        xy_norm = xy * 0.0
    else:
        xy_norm = (xy - (min_xy + size/2.0)) / scale
    if traj.shape[1] == 2:
        return xy_norm.astype(np.float32)
    pen = traj[:, 2:3]
    return np.concatenate([xy_norm, pen], axis=1).astype(np.float32)

def resample_xy(traj: np.ndarray, target_len: int) -> np.ndarray:
    """Resample trajectory to fixed length."""
    if len(traj) == 0:
        ch = traj.shape[1] if traj.ndim == 2 else 3
        return np.zeros((target_len, ch), dtype=np.float32)
    n = traj.shape[0]
    if n == target_len:
        return traj.astype(np.float32)
    old_idx = np.arange(n)
    if n >= target_len:
        new_idx = np.arange(target_len) * (n - 1) / (target_len - 1)
    else:
        new_idx = np.linspace(0, n - 1, target_len)
    
    interp_x = interp1d(old_idx, traj[:, 0], kind='cubic', fill_value="extrapolate")
    interp_y = interp1d(old_idx, traj[:, 1], kind='cubic', fill_value="extrapolate")
    x_new = interp_x(new_idx); y_new = interp_y(new_idx)
    
    if traj.shape[1] >= 3:
        interp_pen = interp1d(old_idx, traj[:, 2], kind='nearest', fill_value="extrapolate")
        pen_new = interp_pen(new_idx)
        return np.column_stack((x_new, y_new, pen_new)).astype(np.float32)
    return np.column_stack((x_new, y_new)).astype(np.float32)

def get_character_freeman_sequence(df_group) -> str:
    """Reconstruct full Freeman code sequence for a character."""
    df_group = df_group.sort_values("strock_number")
    codes = []
    for _, row in df_group.iterrows():
        freeman_str = str(row["strock_label_freeman"])
        codes_str = parse_freeman_string(freeman_str)
        codes.extend(codes_str)
    return ''.join(codes)

# ===== Data Splitting =====
def create_guaranteed_overlap_split(df, min_samples_per_class=2, test_ratio=0.3):
    """Ensure every class appears in both train and test sets"""
    train_list, test_list = [], []
    
    char_counts = df.groupby('hanzi').size().sort_values(ascending=False)
    
    for character in char_counts.index:
        char_data = df[df['hanzi'] == character]
        
        if len(char_data) < min_samples_per_class:
            continue
            
        shuffled = char_data.sample(frac=1, random_state=random_seed)
        n_test = max(1, int(len(shuffled) * test_ratio))
        
        test_list.append(shuffled[:n_test])
        train_list.append(shuffled[n_test:])
    
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    
    return train_df, test_df

# ===== Data Augmentation =====
def shift_trajectory(trajectory):
    out = trajectory.astype(np.float32).copy()
    shift_vector = np.random.uniform(-3, 3, size=2).astype(np.float32)
    out[:, :2] = out[:, :2] + shift_vector
    return out

def rotate_trajectory(trajectory):
    out = trajectory.astype(np.float32).copy()
    center = np.mean(out[:, :2], axis=0)
    rotation_angle = np.random.uniform(-15, 15)
    rotation_angle_rad = math.radians(rotation_angle)
    R = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                  [np.sin(rotation_angle_rad),  np.cos(rotation_angle_rad)]], dtype=np.float32)
    out[:, :2] = ((R @ (out[:, :2] - center).T).T + center)
    return out

def remove_random_points(trajectory):
    num_points = trajectory.shape[0]
    remove_percentage = np.random.uniform(0.05, 0.20)
    remove_count = int(num_points * remove_percentage)
    out = trajectory
    if remove_count > 0 and num_points - remove_count >= 2:
        if np.random.rand() < 0.5:
            out = trajectory[remove_count:, :]
        else:
            out = trajectory[:-remove_count, :]
    return out.astype(np.float32)

def stretch_trajectory(trajectory):
    if len(trajectory) < 3:
        return trajectory.astype(np.float32)
    start_idx = random.randint(0, len(trajectory) - 2)
    end_idx = random.randint(start_idx + 1, len(trajectory) - 1)
    factor = random.uniform(0.5, 1.5)
    out = trajectory.astype(np.float32).copy()
    C = np.mean(out[start_idx:end_idx+1, :2], axis=0)
    S = np.array([[factor, 0], [0, factor]], dtype=np.float32)
    for i in range(start_idx, end_idx + 1):
        out[i, :2] = (S @ (out[i, :2] - C)) + C
    return out

def create_more_positive_pairs(train_samples_xy, train_labels_text, train_freeman, augment_factor=8):
    """Generate augmented samples"""
    augmented_samples, augmented_labels, augmented_freeman = [], [], []
    
    print(f"\nApplying augmentation (factor={augment_factor})...")
    
    for sample, label, freeman in tqdm(zip(train_samples_xy, train_labels_text, train_freeman), 
                                       total=len(train_samples_xy), desc="Augmenting"):
        augmented_samples.append(sample)
        augmented_labels.append(label)
        augmented_freeman.append(freeman)
        
        for _ in range(augment_factor):
            aug_sample = sample.copy()
            
            if np.random.rand() < 0.6:
                aug_sample = shift_trajectory(aug_sample)
            if np.random.rand() < 0.4:
                aug_sample = rotate_trajectory(aug_sample)
            if np.random.rand() < 0.3:
                aug_sample = stretch_trajectory(aug_sample)
            if np.random.rand() < 0.2:
                aug_sample = remove_random_points(aug_sample)
                
            noise = np.random.normal(0, noise_level_small, aug_sample.shape).astype(np.float32)
            aug_sample = aug_sample + noise
            aug_sample = resample_xy(aug_sample, desired_length)
            
            augmented_samples.append(aug_sample)
            augmented_labels.append(label)
            augmented_freeman.append(freeman)
    
    print(f"Augmented: {len(train_samples_xy)} -> {len(augmented_samples)} samples")
    return augmented_samples, augmented_labels, augmented_freeman

# ===== Pair Creation =====
def create_balanced_training_pairs(samples, labels, max_pairs=50000):
    """Create balanced positive/negative pairs"""
    print("\nCreating balanced training pairs...")
    
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[label] for label in labels])
    
    class_samples = {}
    for i, label_id in enumerate(label_ids):
        if label_id not in class_samples:
            class_samples[label_id] = []
        class_samples[label_id].append(i)
    
    positive_pairs = []
    for class_indices in class_samples.values():
        if len(class_indices) > 1:
            positive_pairs.extend(list(combinations(class_indices, 2)))
    
    print(f"Generated {len(positive_pairs)} positive pairs")
    
    n_positive = len(positive_pairs)
    negative_pairs = []
    class_ids = list(class_samples.keys())
    attempts = 0
    max_attempts = min(n_positive * 10, 100000)
    
    while len(negative_pairs) < n_positive and attempts < max_attempts:
        if len(class_ids) < 2:
            break
        c1, c2 = np.random.choice(class_ids, 2, replace=False)
        i1 = np.random.choice(class_samples[c1])
        i2 = np.random.choice(class_samples[c2])
        
        pair = tuple(sorted([i1, i2]))
        if pair not in negative_pairs:
            negative_pairs.append(pair)
        attempts += 1
    
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    all_pairs = positive_pairs + negative_pairs
    pair_labels = [0] * len(positive_pairs) + [1] * len(negative_pairs)
    
    if len(all_pairs) > max_pairs:
        indices = np.random.choice(len(all_pairs), max_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in indices]
        pair_labels = [pair_labels[i] for i in indices]
    
    samples_array = np.array(samples)
    x1 = samples_array[[pair[0] for pair in all_pairs]]
    x2 = samples_array[[pair[1] for pair in all_pairs]]
    y = np.array(pair_labels, dtype=np.float32)
    
    print(f"Final: {len(all_pairs)} pairs ({sum(y==0)} pos, {sum(y==1)} neg)")
    
    return [x1, x2], y

# ===== Build Samples =====
def validate_freeman_codes(codes_str: str) -> bool:
    """
    FIXED: Validate Freeman codes using consistent parsing.
    """
    if not codes_str or str(codes_str).strip() == '' or str(codes_str) == 'nan':
        return False
    
    # Use same parsing method as parse_freeman_string
    parsed = re.findall(r'[0-7]', str(codes_str))
    
    # Filter to valid single-digit codes
    codes = []
    for code_str in parsed:
        if len(code_str) == 1 and int(code_str) <= 7:
            codes.append(int(code_str))
    
    return len(codes) >= 1

def build_samples_from_df(df_in: pd.DataFrame) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
    """Build trajectory samples + Freeman sequences"""
    samples, labels, meta_ids, freeman_seqs = [], [], [], []
    skipped_chars = 0
    
    for (writer, char_nr), g in df_in.groupby(["writer", "character_nr"]):
        g = g.sort_values("strock_number")
        hanzi_label = str(g["hanzi"].iloc[0])
        
        # 1. Collect ALL codes for this character into one sequence
        all_codes = []
        for _, row in g.iterrows():
            freeman_str = str(row["strock_label_freeman"])
            if not validate_freeman_codes(freeman_str):
                continue
            
            # Parse atomic digits
            # This handles "06" -> [0, 6] and "6", "0", "6" -> [6, 0, 6] identically
            atomic_codes = parse_freeman_string(freeman_str)
            for digit in atomic_codes:
                if len(digit) == 1 and digit.isdigit() and int(digit) <= 7:
                    all_codes.append(int(digit))
            
            # Note: We are implicitly connecting separate strokes. 
            # Given we lack relative position data for separate strokes, 
            # assuming connectivity (or 'phantom travel') is the only way to recover shape.
            
        if len(all_codes) < 2:
            skipped_chars += 1
            continue
            
        try:
            # 2. Build ONE trajectory for the whole character
            # This preserves the relative spatial arrangement of the codes
            full_traj = freeman_to_xy(all_codes)
            
            # 3. Normalize and Resample the WHOLE character
            full_traj = normalize_xy(full_traj)
            full_traj = resample_xy(full_traj, desired_length)
            
            samples.append(full_traj)
            labels.append(hanzi_label)
            meta_ids.append(f"{writer}|{char_nr}")
            
            # Reconstruct 'full_freeman' string for reference
            full_freeman = "".join([str(c) for c in all_codes])
            freeman_seqs.append(full_freeman)
            
        except Exception as e:
            # print(f"Error building sample: {e}")
            skipped_chars += 1
            continue
            
        except Exception:
            skipped_chars += 1
            continue
    
    print(f"  Built {len(samples)} samples, skipped {skipped_chars} characters")
    return samples, labels, meta_ids, freeman_seqs

# ===== Loss Function =====
def adaptive_contrastive_loss(y_true, y_pred):
    pos_ratio = tf.reduce_mean(y_true)
    pos_weight = tf.where(pos_ratio > 0.01, (1.0 - pos_ratio) / pos_ratio, 10.0)
    pos_weight = tf.clip_by_value(pos_weight, 1.0, 10.0)
    
    loss_positive = (1 - y_true) * tf.square(y_pred)
    loss_negative = y_true * tf.square(tf.maximum(0.0, margin - y_pred))
    
    weighted_loss = pos_weight * loss_positive + loss_negative
    return tf.reduce_mean(weighted_loss)

# ===== STKNet Layers =====
class SpatialTemporalConvolution(tf.keras.layers.Layer):
    def __init__(self, spatial_kernel_size, spatial_filters, **kwargs):
        super(SpatialTemporalConvolution, self).__init__(**kwargs)
        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_filters = spatial_filters

    def build(self, input_shape):
        self.spatial_conv = tf.keras.layers.Conv2D(self.spatial_filters, self.spatial_kernel_size, padding='same')

    def call(self, spatial_input):
        spatial_features = self.spatial_conv(spatial_input)
        seq = tf.cast(tf.range(1, tf.shape(spatial_input)[1] + 1), tf.float32)
        time_distances = tf.abs(tf.expand_dims(seq, -1) - tf.expand_dims(seq, 0))
        time_distances = tf.expand_dims(tf.expand_dims(time_distances, 0), -1)
        temporal_features = tf.nn.conv2d(
            time_distances,
            tf.ones((self.spatial_kernel_size[0], self.spatial_kernel_size[1], 1, 1)) *
            (1 / (self.spatial_kernel_size[0] * self.spatial_kernel_size[1])),
            strides=(1, 1, 1, 1), padding='SAME'
        )
        return spatial_features * temporal_features

class DistanceMatrixLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        traj1, traj2 = inputs
        traj1 = tf.expand_dims(traj1, axis=2)
        traj2 = tf.expand_dims(traj2, axis=1)
        distances = tf.reduce_sum(tf.square(traj1 - traj2), axis=-1)
        return distances

class GetGlobalForm(Layer):
    def __init__(self, **kwargs):
        super(GetGlobalForm, self).__init__(**kwargs)

    def call(self, inputs):
        matrices = []
        for size in range(5, 13):
            n1 = tf.shape(inputs)[1]
            idx = tf.cast(tf.round(tf.linspace(0.0, tf.cast(n1-1, tf.float32), size)), tf.int32)
            sub_matrix = tf.gather(inputs, idx, axis=1)
            sub_matrix = tf.gather(sub_matrix, idx, axis=2)
            pad_rows = 12 - size
            pad_cols = 12 - size
            sub_matrix_padded = tf.pad(sub_matrix, paddings=[[0, 0], [0, pad_rows], [0, pad_cols]])
            matrices.append(sub_matrix_padded)
        results = tf.stack(matrices, axis=-1)
        return results

class GlobalMaxPooling2D(Layer):
    def __init__(self, **kwargs):
        super(GlobalMaxPooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_max(inputs, axis=[1, 2])

class GlobalInfoProcess(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalInfoProcess, self).__init__(**kwargs)
        self.conv1 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 5))
        self.conv2 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 6))
        self.conv3 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(6, 7))
        self.conv4 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 6))
        self.conv5 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 7))
        self.conv6 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(7, 8))
        self.conv7 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 7))
        self.conv8 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 8))
        self.conv9 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(8, 9))
        self.conv10 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(9, 8))
        self.conv11 = SpatialTemporalConvolution(spatial_filters=5, spatial_kernel_size=(9, 9))
        self.batch_norm = BatchNormalization()
        self.global_max_pooling = GlobalMaxPooling2D()
        self.global_avg_pooling = AveragePooling2D(pool_size=(3, 3))
        self.flatten = Flatten()

    def call(self, inputs):
        x = self.batch_norm(inputs)
        convs = [
            self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), self.conv5(x),
            self.conv6(x), self.conv7(x), self.conv8(x), self.conv9(x), self.conv10(x), self.conv11(x)
        ]
        avg_pooled = [self.global_avg_pooling(c) for c in convs]
        flats = [Flatten()(a) for a in avg_pooled]
        max_pooled = [self.global_max_pooling(c) for c in convs]
        cat1 = self.flatten(tf.concat(max_pooled, axis=-1))
        cat2 = tf.concat(flats, axis=-1)
        return tf.concat([cat1, cat2], axis=-1)

class LocalInfoProcess(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocalInfoProcess, self).__init__(**kwargs)
        self.weight_decay = 0.001
        self.batch_norm = BatchNormalization()
        self.conv33 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 3))
        self.conv34 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 4))
        self.conv35 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(3, 5))
        self.conv43 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 3))
        self.conv44 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 4))
        self.conv45 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 5))
        self.conv46 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(4, 6))
        self.conv53 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 3))
        self.conv54 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 4))
        self.conv55 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 5))
        self.conv56 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(5, 6))
        self.conv64 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 4))
        self.conv65 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 5))
        self.conv66 = SpatialTemporalConvolution(spatial_filters=1, spatial_kernel_size=(6, 6))
        self.flatten = Flatten()
        self.avg_pool33 = AveragePooling2D(pool_size=(3, 3))
        self.global_max_pooling = GlobalMaxPooling2D()

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.batch_norm(x)
        convs = [
            self.conv33(x), self.conv34(x), self.conv35(x), self.conv43(x), self.conv44(x),
            self.conv45(x), self.conv46(x), self.conv53(x), self.conv54(x), self.conv55(x),
            self.conv56(x), self.conv64(x), self.conv65(x), self.conv66(x)
        ]
        aps = [self.avg_pool33(c) for c in convs]
        flats = [Flatten()(a) for a in aps]
        maxps = [self.global_max_pooling(c) for c in convs]
        cat1 = self.flatten(tf.concat(maxps, axis=-1))
        cat2 = tf.concat(flats, axis=-1)
        return tf.concat([cat1, cat2], axis=-1)

class CombineLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)
        self.concat = keras.layers.Concatenate(axis=-1)
        self.fc1 = keras.layers.Dense(units=64, activation='tanh')
        self.fc2 = keras.layers.Dense(units=16, activation='tanh')
        self.fc3 = keras.layers.Dense(units=4, activation='tanh')
        self.fc4 = keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        concatenated = self.concat(inputs)
        x = self.fc1(concatenated)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

# ===== KNN Prediction (FIXED) =====
def class_aware_knn_predict(similarity_matrix, ref_labels, k=None):
    """
    FIXED: Enhanced KNN with proper distance interpretation.
    Model outputs distance (0=similar, 1=dissimilar).
    """
    if similarity_matrix.shape[0] == 0:
        return np.array([], dtype=np.int32)
        
    if k is None:
        unique_classes = len(np.unique(ref_labels))
        samples_per_class = len(ref_labels) / max(1, unique_classes)
        k = min(max(5, int(samples_per_class * 0.5)), 30)
        k = max(3, k)
    
    predictions = []
    
    # DEBUG info for first sample
    if similarity_matrix.shape[0] > 0:
        d_min = np.min(similarity_matrix)
        d_max = np.max(similarity_matrix)
        d_mean = np.mean(similarity_matrix)
        print(f"\nDEBUG: Distance Matrix Stats: Min={d_min:.4f}, Max={d_max:.4f}, Mean={d_mean:.4f}")
        
    for i in range(similarity_matrix.shape[0]):
        # FIXED: Convert distance to similarity
        distances = similarity_matrix[i]
        similarities = 1.0 - distances  # Higher similarity = lower distance
        
        # Get top-k most similar (highest similarity)
        if len(similarities) < k:
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argsort(similarities)[::-1][:k]
        
        votes = ref_labels[top_k_indices]
        weights = similarities[top_k_indices]
        
        # Normalize weights
        weights = np.maximum(weights, 0)
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weights = weights / weight_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        # Weighted voting
        unique_votes = np.unique(votes)
        vote_scores = {}
        
        for vote in unique_votes:
            mask = votes == vote
            vote_scores[vote] = np.sum(weights[mask])
        
        predicted_class = max(vote_scores, key=vote_scores.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

def model_predict_hanzi(test_samples, ref_samples, ref_labels, model, k=None):
    """FIXED: STKNet + Enhanced KNN prediction"""
    n_test = len(test_samples)
    n_ref = len(ref_samples)
    
    if n_test == 0 or n_ref == 0:
        return np.array([], dtype=np.int32), np.zeros((n_test, n_ref), dtype=np.float32)
    
    X1 = np.repeat(np.array(test_samples), n_ref, axis=0)
    X2 = np.tile(np.array(ref_samples), (n_test, 1, 1))
    
    distances = model.predict([X1, X2], batch_size=1024, verbose=0)
    distances_2d = distances.reshape(n_test, n_ref)
    
    predicted_labels = class_aware_knn_predict(distances_2d, ref_labels, k=k)
    
    return predicted_labels, distances_2d

# ===== Character-Level Freeman Evaluation =====
def evaluate_character_level_freeman(test_freeman_seqs, predicted_labels, 
                                     ref_labels, ref_freeman_seqs, id_to_class):
    """Evaluate at character level using Weighted Freeman code sequences."""
    
    class_freeman_map = {}
    for i, label_id in enumerate(ref_labels):
        if label_id not in class_freeman_map:
            class_freeman_map[label_id] = []
        class_freeman_map[label_id].append(ref_freeman_seqs[i])
    
    weighted_edit_distances = []
    perfect_matches = 0
    min_distances_per_sample = []
    avg_distances_per_sample = []
    
    for i, (true_freeman, pred_label_id) in enumerate(zip(test_freeman_seqs, predicted_labels)):
        if pred_label_id not in class_freeman_map:
            weighted_edit_distances.append(1.0)
            min_distances_per_sample.append(1.0)
            avg_distances_per_sample.append(1.0)
            continue
        
        distances_to_class = [
            weighted_freeman_edit_distance(true_freeman, ref_freeman) 
            for ref_freeman in class_freeman_map[pred_label_id]
        ]
        
        weighted_dist_min = min(distances_to_class)
        weighted_dist_avg = np.mean(distances_to_class)
        
        weighted_edit_distances.append(weighted_dist_min)
        min_distances_per_sample.append(weighted_dist_min)
        avg_distances_per_sample.append(weighted_dist_avg)
        
        if weighted_dist_min == 0.0:
            perfect_matches += 1
    
    avg_weighted_distance = np.mean(weighted_edit_distances) if weighted_edit_distances else float('nan')
    avg_of_avg_distances = np.mean(avg_distances_per_sample) if avg_distances_per_sample else float('nan')
    perfect_match_rate = perfect_matches / len(test_freeman_seqs) if test_freeman_seqs else 0
    
    weighted_distances_array = np.array(weighted_edit_distances)
    weighted_acc_01 = np.mean(weighted_distances_array < 0.1) if len(weighted_distances_array) > 0 else 0
    weighted_acc_02 = np.mean(weighted_distances_array < 0.2) if len(weighted_distances_array) > 0 else 0
    weighted_acc_03 = np.mean(weighted_distances_array < 0.3) if len(weighted_distances_array) > 0 else 0
    weighted_acc_05 = np.mean(weighted_distances_array < 0.5) if len(weighted_distances_array) > 0 else 0
    
    return {
        'avg_weighted_distance': avg_weighted_distance,
        'avg_of_avg_distances': avg_of_avg_distances,
        'perfect_match_rate': perfect_match_rate,
        'weighted_edit_distances': weighted_edit_distances,
        'min_distances_per_sample': min_distances_per_sample,
        'avg_distances_per_sample': avg_distances_per_sample,
        'weighted_acc_01': weighted_acc_01,
        'weighted_acc_02': weighted_acc_02,
        'weighted_acc_03': weighted_acc_03,
        'weighted_acc_05': weighted_acc_05
    }

# ===== Helper function =====
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# ===== MAIN EXECUTION =====
print("\n" + "="*60)
print("STARTING MAIN EXECUTION")
print("="*60)

if __name__ == "__main__":
    try:
        # Load data from all Excel files
        print("\n" + "="*80)
        print("LOADING HANZI DATA FROM MULTIPLE FILES")
        print("="*80)
        
        df_list = []
        for excel_file in excel_files:
            print(f"\nLoading: {os.path.basename(excel_file)}")
            try:
                df_temp = pd.read_excel(excel_file)
                df_temp.columns = [c.strip() for c in df_temp.columns]
                df_temp['source_file'] = os.path.basename(excel_file)
                
                print(f"  Loaded {len(df_temp)} samples")
                print(f"  Unique writers: {len(df_temp['writer'].unique())}")
                print(f"  Unique characters: {len(df_temp['hanzi'].unique())}")
                
                df_list.append(df_temp)
            except Exception as e:
                print(f"  ERROR loading {excel_file}: {str(e)}")
                continue
        
        if len(df_list) == 0:
            print("\nERROR: No data could be loaded from Excel files!")
            exit(1)
        
        # Combine all dataframes
        print("\n" + "="*80)
        print("COMBINING ALL DATASETS")
        print("="*80)
        df = pd.concat(df_list, ignore_index=True)
        df["strock_number"] = pd.to_numeric(df["strock_number"], errors="coerce").fillna(0).astype(int)
        
        print(f"\nCombined Dataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Total writers: {len(df['writer'].unique())}")
        print(f"  Total unique characters: {len(df['hanzi'].unique())}")
        
        # Split data
        print("\n" + "="*80)
        print("SPLITTING DATA")
        print("="*80)
        train_df, test_df = create_guaranteed_overlap_split(df, min_samples_per_class=min_samples_per_selected_class, test_ratio=0.2)
        
        print(f"Train df: {len(train_df)} rows")
        print(f"Test df: {len(test_df)} rows")
        
        # Build samples
        print("\nBuilding train samples...")
        train_samples_xy, train_labels_text, train_ids, train_freeman = build_samples_from_df(train_df)
        print("Building test samples...")
        test_samples_xy, test_labels_text, test_ids, test_freeman = build_samples_from_df(test_df)
        
        print(f"\nBuilt {len(train_samples_xy)} train samples, {len(test_samples_xy)} test samples.")
        
        if len(train_samples_xy) == 0 or len(test_samples_xy) == 0:
            print("\n⚠️  ERROR: No valid samples could be built!")
            print("This is likely due to invalid Freeman codes in your data.")
            print("Only 23.8% of your Freeman codes are valid.")
            print("\nPlease check your data collection process.")
            exit(1)
        
        # Limit classes if needed
        if limit_classes_to_top:
            cnt = Counter(train_labels_text)
            valid_classes = [c for c, count in cnt.items() if count >= min_samples_per_selected_class]
            
            if len(valid_classes) == 0:
                print(f"\n⚠️  ERROR: No classes with >= {min_samples_per_selected_class} samples found!")
                print(f"Try reducing min_samples_per_selected_class to 1")
                exit(1)
            
            if len(valid_classes) > limit_classes_to_top:
                top_classes = set([c for c, _ in cnt.most_common() if c in valid_classes][:limit_classes_to_top])
            else:
                top_classes = set(valid_classes)
            
            print(f"\nFiltering to top {len(top_classes)} classes...")
            
            keep_train = [i for i, l in enumerate(train_labels_text) if l in top_classes]
            keep_test = [i for i, l in enumerate(test_labels_text) if l in top_classes]
            
            train_samples_xy = [train_samples_xy[i] for i in keep_train]
            train_labels_text = [train_labels_text[i] for i in keep_train]
            train_freeman = [train_freeman[i] for i in keep_train]
            test_samples_xy = [test_samples_xy[i] for i in keep_test]
            test_labels_text = [test_labels_text[i] for i in keep_test]
            test_freeman = [test_freeman[i] for i in keep_test]
            
            print(f"After filtering: {len(train_samples_xy)} train, {len(test_samples_xy)} test samples")
        
        # Encode labels
        classes = sorted(list(set(train_labels_text + test_labels_text)))
        class_to_id = {c: i for i, c in enumerate(classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        
        train_label_ids = np.array([class_to_id[c] for c in train_labels_text])
        test_label_ids = np.array([class_to_id[c] for c in test_labels_text])
        
        if limit_classes_to_top:
            print(f"Working with {len(valid_classes)} classes")
        
        # AUGMENTATION
        # Increase augmentation to get more variety (crucial since we have few originals)
        augment_factor = 10  # Reduced from 20 to save time (was 10 originally)
        aug_train_samples, aug_train_labels, aug_train_freeman = create_more_positive_pairs(
            train_samples_xy, train_labels_text, train_freeman, augment_factor=augment_factor
        )
        
        # Create pairs for training
        print(f"\nCreating balanced training pairs...")
        train_x, train_y = create_balanced_training_pairs(
            aug_train_samples, aug_train_labels, max_pairs=80000
        )
        
        print(f"Final: {len(train_y)} pairs ({np.sum(train_y==0)} pos, {np.sum(train_y==1)} neg)")
        
        # Build Model
        print(f"\nBuilding STKNet model...")
        
        # Input shape: (N, 3) where 3 is (x, y, pen)
        # Sequence length is dynamic or fixed? Resampled to 128.
        input_shape = (desired_length, 3)
        
        input1 = keras.layers.Input(shape=input_shape)
        input2 = keras.layers.Input(shape=input_shape)
        
        # Shared STKNet
        # In original code, STKNet is a model/layer.
        # We need to compute distance between embeddings.
        
        # Define the Siamese Network
        
        # To ensure we use the same weights, we instantiate the processing layers once
        distance_matrix = DistanceMatrixLayer()([input1, input2])
        # distance_matrix shape: (batch, 128, 128)
        
        get_global_info = GetGlobalForm()
        global_info = get_global_info(distance_matrix)
        global_info_process = GlobalInfoProcess()
        local_info_process = LocalInfoProcess()
        
        global_info_vec = global_info_process(global_info)
        local_info_vec = local_info_process(distance_matrix)
        
        combine_layer = CombineLayer()
        output = combine_layer([local_info_vec, global_info_vec])
        
        model = keras.models.Model(inputs=[input1, input2], outputs=output)
        model.compile(loss=adaptive_contrastive_loss, optimizer=keras.optimizers.Adam(learning_rate=0.001))
        
        model.summary()
        
        # Train
        print("\n" + "="*80)
        print("TRAINING STKNet MODEL")
        print("="*80)
        epochs = 3  # Converged at epoch 3 in previous runs
        batch_size = 256
        
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], 'b-', linewidth=2)
        plt.title("STKNet Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(save_results_root, "training_loss.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Evaluation
        if len(test_samples_xy) > 0:
            print("\n" + "="*80)
            print("EVALUATING MODEL")
            print("="*80)
            
            # Map labels to IDs
            all_labels = sorted(list(set(train_labels_text + test_labels_text)))
            label_to_id = {l: i for i, l in enumerate(all_labels)}
            id_to_class = {i: l for l, i in label_to_id.items()}
            
            train_label_ids = np.array([label_to_id[l] for l in train_labels_text])
            test_label_ids = np.array([label_to_id[l] for l in test_labels_text])
            
            # Use AUGMENTED data as reference for KNN
            # This provides more prototypes per class for better matching
            aug_train_label_ids = np.array([label_to_id[l] for l in aug_train_labels])
            
            # Make predictions
            print("Predicting test samples (STKNet + Enhanced KNN)...")
            enhanced_k = 5 # Default
            
            # Pass augmented samples as reference
            predicted_labels_stknet, dist_matrix_stknet = model_predict_hanzi(
                test_samples_xy, 
                aug_train_samples,  # Use augmented samples
                aug_train_label_ids, # Use corresponding IDs
                model, 
                k=enhanced_k
            )
            
            if len(predicted_labels_stknet) > 0:
                accuracy_stknet = np.mean(predicted_labels_stknet == test_label_ids)
                print(f"\n{'='*60}")
                print(f"STKNet + Enhanced KNN accuracy: {accuracy_stknet:.4f}")
                print(f"{'='*60}")
                
                # DEBUG: Show some predictions
                print("\nDEBUG PREDICTIONS (First 10):")
                for i in range(min(10, len(predicted_labels_stknet))):
                    true_cls = id_to_class[test_label_ids[i]]
                    pred_cls = id_to_class[predicted_labels_stknet[i]]
                    print(f"  Sample {i}: True='{true_cls}' (ID {test_label_ids[i]}), Pred='{pred_cls}' (ID {predicted_labels_stknet[i]})")
                
                # Freeman evaluation
                print("\n" + "="*60)
                print("CHARACTER-LEVEL WEIGHTED FREEMAN CODE EVALUATION")
                print("="*60)
                
                freeman_metrics = evaluate_character_level_freeman(
                    test_freeman, predicted_labels_stknet, 
                    aug_train_label_ids, aug_train_freeman, id_to_class
                )
                
                print(f"\nDirection-Aware Freeman Distance Metrics:")
                print(f"  Average Min Distance: {freeman_metrics['avg_weighted_distance']:.4f}")
                print(f"  Perfect Match Rate: {freeman_metrics['perfect_match_rate']:.4f}")
                print(f"\nCharacter Recognition Accuracy:")
                print(f"  Distance < 0.1: {freeman_metrics['weighted_acc_01']:.4f}")
                print(f"  Distance < 0.2: {freeman_metrics['weighted_acc_02']:.4f}")
                print(f"  Distance < 0.3: {freeman_metrics['weighted_acc_03']:.4f}")
                print(f"  Distance < 0.5: {freeman_metrics['weighted_acc_05']:.4f}")
                
                # Save results
                print(f"\nSaving results to {save_results_root}...")
                
                results_dict = {
                    'classes': classes,
                    'test_labels': test_label_ids,
                    'stknet_predictions': predicted_labels_stknet,
                    'stknet_accuracy': accuracy_stknet,
                    'freeman_metrics': freeman_metrics
                }
                
                for key, value in results_dict.items():
                    if isinstance(value, (list, np.ndarray)):
                        np.save(os.path.join(save_results_root, f"{key}.npy"), value)
                    elif isinstance(value, dict):
                        serializable_dict = make_json_serializable(value)
                        with open(os.path.join(save_results_root, f"{key}.json"), 'w', encoding='utf-8') as f:
                            json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
                    else:
                        with open(os.path.join(save_results_root, f"{key}.txt"), 'w') as f:
                            f.write(str(value))
                
                # Save model
                model.save(os.path.join(save_results_root, "stknet_weighted_freeman_model.keras"))
                
                print("\n" + "="*80)
                print("SCRIPT COMPLETED SUCCESSFULLY!")
                print("="*80)
                print(f"\n⚠️  NOTE on Writer Verification:")
                print(f"  - Current dataset has limited samples per writer per character.")
                print(f"  - Ideally need 3-5 samples per character per writer for robust verification.")
                
            else:
                print("No predictions generated.")
        else:
            print("No valid samples for evaluation.")
            
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR OCCURRED: {type(e).__name__}")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}")
