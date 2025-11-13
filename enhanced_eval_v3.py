# -*- coding: utf-8 -*-


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

print("=" * 60)
print("SCRIPT INITIALIZATION STARTED")
print("=" * 60)

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ===== Configuration =====
excel_path = "/Users/ballu_macbookpro/Desktop/Thesis/strokeLablesForPot1.0train.xlsx"
save_results_root = os.path.expanduser("~/Downloads/results_stknet_hanzi_freeman_weighted")
os.makedirs(save_results_root, exist_ok=True)

print(f"\nChecking data file: {excel_path}")
print(f"File exists: {os.path.exists(excel_path)}")
if not os.path.exists(excel_path):
    print("ERROR: Excel file not found! Please check the path.")
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

limit_classes_to_top = 50

# ===== Direction-Aware Freeman Code Distance =====
def freeman_angular_difference(code1: int, code2: int) -> float:
    """
    Compute angular difference between two Freeman codes (0-7).
    Returns normalized difference [0, 1] based on angular distance.
    
    Freeman code directions:
      3  2  1
       ↖ ↑ ↗
    4 ← · → 0
       ↙ ↓ ↘
      5  6  7
    
    Examples:
        0 → 1 (adjacent, 45°):  0.125
        0 → 4 (opposite, 180°): 1.0
        0 → 0 (same):           0.0
    """
    # Angular difference in Freeman code units (each unit = 45°)
    diff = abs(code1 - code2)
    
    # Handle wrap-around (e.g., 0 and 7 are adjacent)
    if diff > 4:
        diff = 8 - diff
    
    # Normalize by maximum angular distance (4 units = 180°)
    return diff / 4.0


def weighted_freeman_edit_distance(seq1: str, seq2: str) -> float:
    """
    Compute weighted edit distance between Freeman code sequences.
    Substitution cost is proportional to angular difference between directions.
    
    This is more meaningful than standard Levenshtein distance because it
    considers the geometric similarity of Freeman codes (directional movements).
    
    Args:
        seq1, seq2: Freeman code strings (digits 0-7)
    
    Returns:
        Normalized weighted edit distance [0, 1]
    """
    # Clean sequences - remove non-digit characters and invalid codes
    s1 = ''.join(c for c in str(seq1) if c.isdigit() and int(c) <= 7)
    s2 = ''.join(c for c in str(seq2) if c.isdigit() and int(c) <= 7)
    
    if not s1 and not s2:
        return 0.0
    
    n, m = len(s1), len(s2)
    
    # Initialize DP table
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    
    # Base cases: insertions/deletions have cost 1.0
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    
    # Fill DP table with weighted costs
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            code1 = int(s1[i-1])
            code2 = int(s2[j-1])
            
            if code1 == code2:
                # Exact match: no cost
                substitution_cost = 0.0
            else:
                # Substitution cost based on angular difference
                # Adjacent directions (45° apart) cost 0.25
                # Opposite directions (180° apart) cost 1.0
                substitution_cost = freeman_angular_difference(code1, code2)
            
            dp[i][j] = min(
                dp[i-1][j] + 1.0,                    # Deletion
                dp[i][j-1] + 1.0,                    # Insertion
                dp[i-1][j-1] + substitution_cost      # Substitution (weighted)
            )
    
    # Normalize by maximum possible distance
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
    return re.findall(r'\d+', s)

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
def create_guaranteed_overlap_split(df, min_samples_per_class=3, test_ratio=0.3):
    """Ensure every class appears in both train and test sets"""
    train_list, test_list = [], []
    
    print("\nAnalyzing character distribution...")
    char_counts = df.groupby('hanzi').size().sort_values(ascending=False)
    print(f"Total characters: {len(char_counts)}")
    print(f"Characters with >= {min_samples_per_class} samples: {sum(char_counts >= min_samples_per_class)}")
    
    for character in char_counts.index:
        char_data = df[df['hanzi'] == character]
        
        if len(char_data) < min_samples_per_class:
            continue
            
        shuffled = char_data.sample(frac=1, random_state=random_seed)
        n_test = max(1, int(len(shuffled) * test_ratio))
        
        test_list.append(shuffled[:n_test])
        train_list.append(shuffled[n_test:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    print(f"\nSplit result:")
    print(f"  Train: {len(train_df)} samples, {len(train_df['hanzi'].unique())} classes")
    print(f"  Test: {len(test_df)} samples, {len(test_df['hanzi'].unique())} classes")
    print(f"  Class overlap: {len(set(train_df['hanzi'].unique()) & set(test_df['hanzi'].unique()))}")
    
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
    if not codes_str or str(codes_str).strip() == '' or str(codes_str) == 'nan':
        return False
    codes = [int(c) for c in str(codes_str) if c.isdigit() and int(c) <= 7]
    return len(codes) >= 2

def build_samples_from_df(df_in: pd.DataFrame) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
    """Build trajectory samples + Freeman sequences"""
    samples, labels, meta_ids, freeman_seqs = [], [], [], []
    skipped_chars = 0
    
    print("\nBuilding samples from DataFrame...")
    
    for (writer, char_nr), g in tqdm(df_in.groupby(["writer", "character_nr"]), desc="Processing"):
        g = g.sort_values("strock_number")
        hanzi_label = str(g["hanzi"].iloc[0])
        traj_parts = []
        
        full_freeman = get_character_freeman_sequence(g)
        
        for _, row in g.iterrows():
            freeman_str = str(row["strock_label_freeman"])
            
            if not validate_freeman_codes(freeman_str):
                continue
                
            codes_str = parse_freeman_string(freeman_str)
            codes = [int(c) for c in codes_str if c.isdigit() and int(c) <= 7]
            
            if len(codes) >= 2:
                try:
                    stroke = freeman_to_xy(codes)
                    if len(stroke) > 1:
                        traj_parts.append(stroke)
                except Exception:
                    continue
        
        if len(traj_parts) == 0:
            skipped_chars += 1
            continue
        
        seq = []
        for s_idx, stroke in enumerate(traj_parts):
            if s_idx > 0 and len(seq) > 0:
                last_xy = seq[-1][:2]
                sep = np.array([last_xy[0], last_xy[1], 0.0], dtype=np.float32)
                seq.append(sep)
            seq.extend(stroke.tolist())
        
        if len(seq) < 2:
            skipped_chars += 1
            continue
        
        try:
            char_traj = np.array(seq, dtype=np.float32)
            char_traj = normalize_xy(char_traj)
            char_traj = resample_xy(char_traj, desired_length)
            
            samples.append(char_traj)
            labels.append(hanzi_label)
            meta_ids.append(f"{writer}|{char_nr}")
            freeman_seqs.append(full_freeman)
            
        except Exception:
            skipped_chars += 1
            continue
    
    if skipped_chars > 0:
        print(f"Skipped {skipped_chars} characters")
    
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

# ===== KNN Prediction =====
def class_aware_knn_predict(similarity_matrix, ref_labels, k=None):
    """Enhanced KNN with class-aware distance weighting"""
    if similarity_matrix.shape[0] == 0:
        return np.array([], dtype=np.int32)
        
    if k is None:
        unique_classes = len(np.unique(ref_labels))
        samples_per_class = len(ref_labels) / max(1, unique_classes)
        k = min(max(3, int(samples_per_class * 0.5)), similarity_matrix.shape[1] // 2)
        k = max(1, k)
    
    print(f"Using k={k} for KNN voting")
    
    predictions = []
    for i in range(similarity_matrix.shape[0]):
        similarities = 1.0 - similarity_matrix[i]
        
        if len(similarities) < k:
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argsort(similarities)[::-1][:k]
        
        votes = ref_labels[top_k_indices]
        weights = similarities[top_k_indices]
        weights = np.maximum(weights, 1e-8)
        
        unique_votes = np.unique(votes)
        vote_scores = {}
        
        for vote in unique_votes:
            mask = votes == vote
            vote_scores[vote] = np.sum(weights[mask])
        
        predicted_class = max(vote_scores, key=vote_scores.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

def model_predict_hanzi(test_samples, ref_samples, ref_labels, model, k=None):
    """STKNet + Enhanced KNN prediction"""
    n_test = len(test_samples)
    n_ref = len(ref_samples)
    
    if n_test == 0 or n_ref == 0:
        print(f"Warning: Skipping prediction (n_test={n_test}, n_ref={n_ref}).")
        return np.array([], dtype=np.int32), np.zeros((n_test, n_ref), dtype=np.float32)
    
    print(f"\nComputing STKNet similarities for {n_test} test vs {n_ref} reference samples...")
    
    X1 = np.repeat(np.array(test_samples), n_ref, axis=0)
    X2 = np.tile(np.array(ref_samples), (n_test, 1, 1))
    
    distances = model.predict([X1, X2], batch_size=1024, verbose=1)
    distances_2d = distances.reshape(n_test, n_ref)
    
    predicted_labels = class_aware_knn_predict(distances_2d, ref_labels, k=k)
    
    return predicted_labels, distances_2d

# ===== Character-Level Freeman Evaluation =====
def evaluate_character_level_freeman(test_freeman_seqs, predicted_labels, 
                                     ref_labels, ref_freeman_seqs, id_to_class):
    """
    Evaluate at character level using Weighted Freeman code sequences.
    Compare against ALL training samples of predicted class.
    """
    print("\n" + "="*60)
    print("CHARACTER-LEVEL WEIGHTED FREEMAN CODE EVALUATION")
    print("="*60)
    
    # Build mapping from class_id to ALL Freeman sequences in that class
    class_freeman_map = {}
    for i, label_id in enumerate(ref_labels):
        if label_id not in class_freeman_map:
            class_freeman_map[label_id] = []
        class_freeman_map[label_id].append(ref_freeman_seqs[i])
    
    # Statistics about reference samples per class
    samples_per_class = {k: len(v) for k, v in class_freeman_map.items()}
    avg_samples = np.mean(list(samples_per_class.values()))
    print(f"\n Reference Dataset Statistics:")
    print(f"   Total classes: {len(class_freeman_map)}")
    print(f"   Avg samples per class: {avg_samples:.2f}")
    print(f"   Min samples per class: {min(samples_per_class.values())}")
    print(f"   Max samples per class: {max(samples_per_class.values())}")
    
    weighted_edit_distances = []
    perfect_matches = 0
    min_distances_per_sample = []
    avg_distances_per_sample = []
    
    strategy_differences = 0
    
    for i, (true_freeman, pred_label_id) in enumerate(zip(test_freeman_seqs, predicted_labels)):
        if pred_label_id not in class_freeman_map:
            weighted_edit_distances.append(1.0)
            min_distances_per_sample.append(1.0)
            avg_distances_per_sample.append(1.0)
            continue
        
        # Compare to ALL training samples of predicted class
        distances_to_class = [
            weighted_freeman_edit_distance(true_freeman, ref_freeman) 
            for ref_freeman in class_freeman_map[pred_label_id]
        ]
        
        # Use minimum distance (best match within class)
        weighted_dist_min = min(distances_to_class)
        weighted_dist_avg = np.mean(distances_to_class)
        
        # For comparison: what would the old method (first sample only) give?
        weighted_dist_first = distances_to_class[0]
        
        if abs(weighted_dist_min - weighted_dist_first) > 0.01:
            strategy_differences += 1
        
        # Store results
        weighted_edit_distances.append(weighted_dist_min)
        min_distances_per_sample.append(weighted_dist_min)
        avg_distances_per_sample.append(weighted_dist_avg)
        
        if weighted_dist_min == 0.0:
            perfect_matches += 1
    
    # Compute metrics
    avg_weighted_distance = np.mean(weighted_edit_distances) if weighted_edit_distances else float('nan')
    avg_of_avg_distances = np.mean(avg_distances_per_sample) if avg_distances_per_sample else float('nan')
    perfect_match_rate = perfect_matches / len(test_freeman_seqs) if test_freeman_seqs else 0
    
    # Calculate accuracy at different thresholds
    weighted_distances_array = np.array(weighted_edit_distances)
    weighted_acc_01 = np.mean(weighted_distances_array < 0.1) if len(weighted_distances_array) > 0 else 0
    weighted_acc_02 = np.mean(weighted_distances_array < 0.2) if len(weighted_distances_array) > 0 else 0
    weighted_acc_03 = np.mean(weighted_distances_array < 0.3) if len(weighted_distances_array) > 0 else 0
    weighted_acc_05 = np.mean(weighted_distances_array < 0.5) if len(weighted_distances_array) > 0 else 0
    
    print(f"\n Distance Computation Strategy:")
    print(f"   Using: Minimum distance to ANY training sample in predicted class")
    print(f"   Samples where min ≠ first[0]: {strategy_differences}/{len(test_freeman_seqs)} ({100*strategy_differences/len(test_freeman_seqs):.1f}%)")
    
    print(f"\n Direction-Aware Freeman Distance Metrics:")
    print(f"   Average Min Distance: {avg_weighted_distance:.4f}")
    print(f"   Average of Avg Distances: {avg_of_avg_distances:.4f}")
    print(f"   Perfect Match Rate: {perfect_match_rate:.4f} ({perfect_matches}/{len(test_freeman_seqs)})")
    
    print(f"\n Character Recognition Accuracy (using min distance):")
    print(f"   Distance < 0.1: {weighted_acc_01:.4f} ({int(weighted_acc_01*len(test_freeman_seqs))}/{len(test_freeman_seqs)} samples)")
    print(f"   Distance < 0.2: {weighted_acc_02:.4f} ({int(weighted_acc_02*len(test_freeman_seqs))}/{len(test_freeman_seqs)} samples)")
    print(f"   Distance < 0.3: {weighted_acc_03:.4f} ({int(weighted_acc_03*len(test_freeman_seqs))}/{len(test_freeman_seqs)} samples)")
    print(f"   Distance < 0.5: {weighted_acc_05:.4f} ({int(weighted_acc_05*len(test_freeman_seqs))}/{len(test_freeman_seqs)} samples)")
    
    # Distribution analysis
    print(f"\n Distance Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(weighted_distances_array, p)
        print(f"   {p}th percentile: {val:.4f}")
    
    print("="*60)
    
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
        'weighted_acc_05': weighted_acc_05,
        'strategy_differences': strategy_differences,
        'samples_per_class': samples_per_class
    }


# ===== Additional Character-Level Evaluation with Hyphen Removal =====
def evaluate_character_level_detailed(test_freeman, predicted_labels, 
                                       ref_labels, ref_freeman, id_to_class):
    """
    Additional character-level evaluation with hyphens removed.
    
    This evaluation:
    1. Removes hyphens from Freeman sequences as specified
    2. Compares predicted vs ground-truth at character level
    3. Uses normalized edit distance (distance / max_length)
    4. Provides detailed accuracy metrics at multiple thresholds
    """
    print("\n" + "="*60)
    print("DETAILED CHARACTER-LEVEL EVALUATION")
    print("(Freeman sequences with hyphens removed)")
    print("="*60)
    
    # Build class -> Freeman mapping (hyphens removed)
    class_freeman_map = {}
    for i, label_id in enumerate(ref_labels):
        if label_id not in class_freeman_map:
            class_freeman_map[label_id] = []
        clean_freeman = ref_freeman[i].replace('-', '')
        class_freeman_map[label_id].append(clean_freeman)
    
    print(f"\n Reference Data Statistics:")
    print(f"   Total classes: {len(class_freeman_map)}")
    samples_per_class = [len(v) for v in class_freeman_map.values()]
    print(f"   Avg samples per class: {np.mean(samples_per_class):.2f}")
    
    results = []
    perfect_matches = 0
    
    for idx, (true_freeman, pred_label_id) in enumerate(zip(test_freeman, predicted_labels)):
        # Remove hyphens from test sequence
        true_freeman_clean = true_freeman.replace('-', '')
        
        if pred_label_id in class_freeman_map:
            ref_freemans = class_freeman_map[pred_label_id]
            
            # Compute normalized distances to all reference samples
            distances = [
                weighted_freeman_edit_distance(true_freeman_clean, ref_f)
                for ref_f in ref_freemans
            ]
            
            best_dist = min(distances)
            avg_dist = np.mean(distances)
        else:
            best_dist = 1.0
            avg_dist = 1.0
        
        results.append({
            'sample_idx': idx,
            'pred_label': id_to_class.get(pred_label_id, 'UNKNOWN'),
            'best_distance': best_dist,
            'avg_distance': avg_dist,
            'true_length': len(true_freeman_clean)
        })
        
        if best_dist == 0.0:
            perfect_matches += 1
    
    # Aggregate metrics
    distances = np.array([r['best_distance'] for r in results])
    
    print(f"\n Character-Level Metrics (Hyphens Removed):")
    print(f"   Total test characters: {len(results)}")
    print(f"   Perfect matches (distance = 0.0): {perfect_matches}/{len(results)} ({perfect_matches/len(results):.4f})")
    print(f"   Mean normalized distance: {np.mean(distances):.4f}")
    print(f"   Median normalized distance: {np.median(distances):.4f}")
    print(f"   Std deviation: {np.std(distances):.4f}")
    
    print(f"\n Recognition Accuracy by Distance Threshold:")
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        acc = np.mean(distances <= threshold)
        n = int(acc * len(results))
        print(f"   Distance ≤ {threshold:.1f}: {acc:.4f} ({n}/{len(results)} characters)")
    
    print(f"\n Distance Normalization:")
    print(f"   Normalization formula: raw_distance / max(len_seq1, len_seq2)")
    print(f"   All distances in [0,1]: {np.all((distances >= 0) & (distances <= 1))}")
    print(f"   Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
    
    print(f"\n Distance Distribution (Percentiles):")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(distances, p)
        print(f"   {p}th percentile: {val:.4f}")
    

    
    return {
        'distances': distances,
        'results': results,
        'perfect_matches': perfect_matches,
        'perfect_match_rate': perfect_matches / len(results) if results else 0.0,
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'accuracy_at_00': float(np.mean(distances == 0.0)),
        'accuracy_at_01': float(np.mean(distances <= 0.1)),
        'accuracy_at_02': float(np.mean(distances <= 0.2)),
        'accuracy_at_03': float(np.mean(distances <= 0.3)),
        'accuracy_at_05': float(np.mean(distances <= 0.5))
    }


# ===== MAIN EXECUTION =====
print("\n" + "="*60)
print("STARTING MAIN EXECUTION")
print("="*60)

if __name__ == "__main__":
    try:
        # Load data
        print("\nLoading Hanzi data...")
        df = pd.read_excel(excel_path)
        df.columns = [c.strip() for c in df.columns]
        df["strock_number"] = pd.to_numeric(df["strock_number"], errors="coerce").fillna(0).astype(int)
        
        print(f"Loaded {len(df)} samples from {len(df['writer'].unique())} writers")
        print(f"Total unique characters: {len(df['hanzi'].unique())}")
        
        # Split data
        train_df, test_df = create_guaranteed_overlap_split(df, min_samples_per_class=3, test_ratio=0.3)
        
        # Build samples
        train_samples_xy, train_labels_text, train_ids, train_freeman = build_samples_from_df(train_df)
        test_samples_xy, test_labels_text, test_ids, test_freeman = build_samples_from_df(test_df)
        
        print(f"\nBuilt {len(train_samples_xy)} train samples, {len(test_samples_xy)} test samples.")
        
        # Verify class overlap
        train_classes = set(train_labels_text)
        test_classes = set(test_labels_text)
        overlap = train_classes & test_classes
        print(f"Class overlap: {len(overlap)} common classes")
        
        # Limit classes if needed
        if limit_classes_to_top and len(train_classes) > limit_classes_to_top:
            cnt = Counter(train_labels_text)
            top_classes = set([c for c, _ in cnt.most_common(limit_classes_to_top)])
            
            keep_train = [i for i, l in enumerate(train_labels_text) if l in top_classes]
            keep_test = [i for i, l in enumerate(test_labels_text) if l in top_classes]
            
            train_samples_xy = [train_samples_xy[i] for i in keep_train]
            train_labels_text = [train_labels_text[i] for i in keep_train]
            train_freeman = [train_freeman[i] for i in keep_train]
            test_samples_xy = [test_samples_xy[i] for i in keep_test]
            test_labels_text = [test_labels_text[i] for i in keep_test]
            test_freeman = [test_freeman[i] for i in keep_test]
            
            print(f"\nLimited to top {limit_classes_to_top} classes")
            print(f"Samples: {len(train_samples_xy)} train, {len(test_samples_xy)} test")
        
        # Encode labels
        classes = sorted(list(set(train_labels_text + test_labels_text)))
        class_to_id = {c: i for i, c in enumerate(classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}
        
        train_label_ids = np.array([class_to_id[c] for c in train_labels_text])
        test_label_ids = np.array([class_to_id[c] for c in test_labels_text])
        
        num_classes = len(classes)
        print(f"\nWorking with {num_classes} classes")
        
        # Augmentation
        augmented_samples, augmented_labels, augmented_freeman = create_more_positive_pairs(
            train_samples_xy, train_labels_text, train_freeman, augment_factor=2
        )
        
        # Convert to XY-only
        augmented_samples_2d = [s[:, :2] for s in augmented_samples]
        
        # Create training pairs
        train_x, train_y = create_balanced_training_pairs(
            augmented_samples_2d, augmented_labels, max_pairs=50000
        )
        
        # Build STKNet model
        print("\nBuilding STKNet model...")
        trajectory_length = desired_length
        num_dimensions = 2
        
        input1 = keras.layers.Input(shape=(trajectory_length, num_dimensions))
        input2 = keras.layers.Input(shape=(trajectory_length, num_dimensions))
        distance_layer = DistanceMatrixLayer()
        distance_matrix = distance_layer([input1, input2])
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
        
        print("\nSTKNet Model Summary:")
        model.summary()
        
        # Train STKNet
        print("\nTraining STKNet model...")
        epochs = 2
        batch_size = 512
        
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], 'b-', linewidth=2)
        plt.title("STKNet Training Loss (Weighted Freeman Distance)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(save_results_root, "training_loss.png"), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Prepare reference and test data
        ref_samples_2d = [s[:, :2] for s in train_samples_xy]
        ref_labels = train_label_ids
        ref_freeman = train_freeman
        test_samples_2d = [s[:, :2] for s in test_samples_xy]
        
        # Evaluation
        if len(test_samples_2d) > 0 and len(ref_samples_2d) > 0:
            # STKNet + KNN prediction
            predicted_labels_stknet, similarity_matrix = model_predict_hanzi(
                test_samples_2d, ref_samples_2d, ref_labels, model
            )
            
            if len(predicted_labels_stknet) > 0:
                accuracy_stknet = np.mean(predicted_labels_stknet == test_label_ids)
                print(f"\n{'='*60}")
                print(f"STKNet + Enhanced KNN accuracy: {accuracy_stknet:.4f}")
                print(f"{'='*60}")
                
                # Character-level Weighted Freeman evaluation
                freeman_metrics = evaluate_character_level_freeman(
                    test_freeman, predicted_labels_stknet, 
                    ref_labels, ref_freeman, id_to_class
                )
                
                # Additional evaluation with hyphens removed
                detailed_metrics = evaluate_character_level_detailed(
                    test_freeman, predicted_labels_stknet,
                    ref_labels, ref_freeman, id_to_class
                )
                
                # Save results
                print(f"\nSaving results to {save_results_root}...")
                
                import json
                
                # Helper function to convert numpy/python types to JSON-serializable format
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
                
                results_dict = {
                    'classes': classes,
                    'test_labels': test_label_ids,
                    'stknet_predictions': predicted_labels_stknet,
                    'similarity_matrix': similarity_matrix,
                    'stknet_accuracy': accuracy_stknet,
                    'training_history': history.history,
                    'freeman_metrics': freeman_metrics,
                    'detailed_metrics': detailed_metrics
                }
                
                for key, value in results_dict.items():
                    if isinstance(value, (list, np.ndarray)):
                        np.save(os.path.join(save_results_root, f"{key}.npy"), value)
                    elif isinstance(value, dict):
                        # Convert to JSON-serializable format
                        serializable_dict = make_json_serializable(value)
                        with open(os.path.join(save_results_root, f"{key}.json"), 'w', encoding='utf-8') as f:
                            json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
                    else:
                        with open(os.path.join(save_results_root, f"{key}.txt"), 'w') as f:
                            f.write(str(value))
                
                # Save model
                model.save(os.path.join(save_results_root, "stknet_weighted_freeman_model.keras"))
                
                # Final summary
                print(f"\n{'='*60}")
                print("FINAL RESULTS SUMMARY")
                print(f"{'='*60}")
                print(f"Dataset: {num_classes} Hanzi classes")
                print(f"Train: {len(train_samples_xy)} original -> {len(augmented_samples)} augmented")
                print(f"Test: {len(test_samples_xy)} samples")
                print(f"Training pairs: {len(train_x[0])} balanced pairs")
                print(f"\n STKNet Accuracy: {accuracy_stknet:.4f}")
                print(f"\n Weighted Freeman Metrics (with hyphens):")
                print(f"   Avg min distance: {freeman_metrics['avg_weighted_distance']:.4f}")
                print(f"   Perfect match rate: {freeman_metrics['perfect_match_rate']:.4f}")
                print(f"   Accuracy (< 0.3): {freeman_metrics['weighted_acc_03']:.4f}")
                print(f"\n Detailed Metrics (hyphens removed):")
                print(f"   Mean distance: {detailed_metrics['mean_distance']:.4f}")
                print(f"   Accuracy (≤ 0.1): {detailed_metrics['accuracy_at_01']:.4f}")
                print(f"   Accuracy (≤ 0.2): {detailed_metrics['accuracy_at_02']:.4f}")
                print(f"   Accuracy (≤ 0.3): {detailed_metrics['accuracy_at_03']:.4f}")
                print(f"{'='*60}")
                
                print("\n Script completed successfully!")
                print(f"Results saved in: {save_results_root}")
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
