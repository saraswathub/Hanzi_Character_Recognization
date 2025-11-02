# -*- coding: utf-8 -*-
"""
STKNet + Edit Distance for Hanzi Character Recognition
"""

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
import matplotlib.pyplot as plt
import matplotlib as mpl

# Other imports
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from difflib import SequenceMatcher  # For edit distance computation

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ===== Configuration =====
excel_path = "/Users/ballu_macbookpro/Desktop/Thesis/strokeLablesForPot1.0train.xlsx"
save_results_root = os.path.expanduser("~/Downloads/results_stknet_hanzi_edit")
os.makedirs(save_results_root, exist_ok=True)

desired_length = 25  # Same as original STKNET
margin = 0.3
noise_level_small = 1 / 3
random_seed = 1000

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Configure matplotlib for CJK characters
try:
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

limit_classes_to_top = 50  # For development

# ===== Freeman Chain Code Utilities =====
FREEMAN_8 = {
    0: (1, 0),   1: (1, 1),   2: (0, 1),   3: (-1, 1),
    4: (-1, 0),  5: (-1,-1),  6: (0,-1),  7: (1,-1)
}

def parse_freeman_string(s: str) -> List[str]:
    """Extract Freeman codes from strings, preserving structure for edit distance."""
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
    
    # Cubic interpolation for XY
    interp_x = interp1d(old_idx, traj[:, 0], kind='cubic', fill_value="extrapolate")
    interp_y = interp1d(old_idx, traj[:, 1], kind='cubic', fill_value="extrapolate")
    x_new = interp_x(new_idx); y_new = interp_y(new_idx)
    
    if traj.shape[1] >= 3:
        # Nearest neighbor for pen
        interp_pen = interp1d(old_idx, traj[:, 2], kind='nearest', fill_value="extrapolate")
        pen_new = interp_pen(new_idx)
        return np.column_stack((x_new, y_new, pen_new)).astype(np.float32)
    return np.column_stack((x_new, y_new)).astype(np.float32)

# ===== NEW: Edit Distance Functions =====
def compute_freeman_edit_distance(codes1: List[str], codes2: List[str], normalize: bool = True) -> float:
    """
    Compute normalized edit distance between two Freeman code sequences.
    
    Args:
        codes1: First sequence of Freeman codes
        codes2: Second sequence of Freeman codes
        normalize: If True, normalize the distance by the length of the longer sequence
        
    Returns:
        float: Normalized edit distance between 0 (identical) and 1 (completely different)
    """
    if not codes1 and not codes2:
        return 0.0
    if not codes1 or not codes2:
        return 1.0
    
    # Convert to strings for SequenceMatcher
    str1 = ''.join(codes1)
    str2 = ''.join(codes2)
    
    # Use SequenceMatcher to compute edit operations
    matcher = SequenceMatcher(None, str1, str2)
    
    if not normalize:
        # Raw edit distance (number of operations)
        return 1.0 - matcher.ratio()
    
    # Normalized edit distance
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
        
    # Calculate operations and normalize
    ops = matcher.get_opcodes()
    edit_distance = sum(max(i2-i1, j2-j1) for op, i1, i2, j1, j2 in ops 
                       if op != 'equal')
    
    return min(1.0, edit_distance / max_len)

def compute_stroke_level_edit_distance(strokes1: List[str], strokes2: List[str]) -> float:
    """
    Compute edit distance at stroke level (comparing stroke sequences).
    Each stroke is treated as a single unit.
    """
    if not strokes1 and not strokes2:
        return 0.0
    if not strokes1 or not strokes2:
        return 1.0
    
    # Use SequenceMatcher on stroke sequences
    similarity = SequenceMatcher(None, strokes1, strokes2).ratio()
    distance = 1.0 - similarity
    
    return distance

def compute_combined_edit_distance(strokes1: List[str], strokes2: List[str], 
                                 alpha=0.7) -> float:
    """
    Combine character-level and stroke-level edit distances.
    alpha: weight for character-level distance (1-alpha for stroke-level)
    """
    char_distance = compute_freeman_edit_distance(strokes1, strokes2)
    stroke_distance = compute_stroke_level_edit_distance(strokes1, strokes2)
    
    combined = alpha * char_distance + (1 - alpha) * stroke_distance
    return combined

# ===== Data Augmentation Function =====
def shift_trajectory(trajectory):
    """Shift trajectory by random vector."""
    out = trajectory.astype(np.float32).copy()
    shift_vector = np.random.uniform(-3, 3, size=2).astype(np.float32)
    out[:, :2] = out[:, :2] + shift_vector
    return out

def rotate_trajectory(trajectory):
    """Rotate trajectory around last point."""
    out = trajectory.astype(np.float32).copy()
    pivot_point = out[-1, :2]
    rotation_angle = np.random.uniform(5, 15)
    rotation_angle_rad = math.radians(rotation_angle)
    R = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                  [np.sin(rotation_angle_rad),  np.cos(rotation_angle_rad)]], dtype=np.float32)
    out[:, :2] = ((R @ (out[:, :2] - pivot_point).T).T + pivot_point)
    return out

def remove_random_points(trajectory):
    """Remove random points from trajectory."""
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
    """Apply random scaling to trajectory segment."""
    if len(trajectory) < 3:
        return trajectory.astype(np.float32)
    start_idx = random.randint(0, len(trajectory) - 2)
    end_idx = random.randint(start_idx + 1, len(trajectory) - 1)
    factor = random.uniform(0.2, 1.8)
    out = trajectory.astype(np.float32).copy()
    C = np.mean(out[start_idx:end_idx+1, :2], axis=0)
    S = np.array([[factor, 0], [0, factor]], dtype=np.float32)
    for i in range(start_idx, end_idx + 1):
        out[i, :2] = (S @ (out[i, :2] - C)) + C
    return out

# ===== Data Loading and Processing =====
def build_samples_from_df(df_in: pd.DataFrame) -> Tuple[List[np.ndarray], List[str], List[str], List[List[str]]]:
    """
    Build trajectory samples from DataFrame.
    Returns: (samples, labels, meta_ids, freeman_codes)
    """
    samples, labels, meta_ids, freeman_codes = [], [], [], []
    
    for (writer, char_nr), g in df_in.groupby(["writer", "character_nr"]):
        g = g.sort_values("strock_number")
        hanzi_label = str(g["hanzi"].iloc[0])
        traj_parts = []
        char_freeman_codes = []
        
        for _, row in g.iterrows():
            codes_str = str(row["strock_label_freeman"])
            codes = parse_freeman_string(codes_str)
            char_freeman_codes.extend(codes)  # Flatten all stroke codes
            
            if len(codes) == 0:
                continue
            
            # Convert to trajectory for visualization/LSTM
            codes_int = [int(c) for c in codes if c.isdigit()]
            if codes_int:
                stroke = freeman_to_xy(codes_int)
                traj_parts.append(stroke)
        
        if len(traj_parts) == 0:
            continue
        
        # Concatenate strokes with pen-up separators
        seq = []
        for s_idx, stroke in enumerate(traj_parts):
            if s_idx > 0 and len(seq) > 0:
                last_xy = seq[-1][:2]
                sep = np.array([last_xy[0], last_xy[1], 0.0], dtype=np.float32)
                seq.append(sep)
            seq.extend(stroke.tolist())
        
        char_traj = np.array(seq, dtype=np.float32)
        char_traj = normalize_xy(char_traj)
        char_traj = resample_xy(char_traj, desired_length)
        
        samples.append(char_traj)
        labels.append(hanzi_label)
        meta_ids.append(f"{writer}|{char_nr}")
        freeman_codes.append(char_freeman_codes)
    
    return samples, labels, meta_ids, freeman_codes

def createData_hanzi_edit_distance(train_samples_xy, train_labels_text, train_freeman_codes):
    """
    MAIN CHANGE: Modified createData function to use edit distance labels instead of binary labels.
    """
    # Convert to numpy for easier manipulation
    base_samples = [s[:, :2] if s.shape[1] > 2 else s for s in train_samples_xy]
    
    # Apply augmentations (same as original)
    shifted = [resample_xy(shift_trajectory(s), desired_length)[:, :2] for s in train_samples_xy]
    rotated = [resample_xy(rotate_trajectory(s), desired_length)[:, :2] for s in train_samples_xy]
    cropped = [resample_xy(remove_random_points(s), desired_length)[:, :2] for s in train_samples_xy]
    stretched = [resample_xy(stretch_trajectory(s), desired_length)[:, :2] for s in train_samples_xy]
    
    # Add noise (same as original)
    def add_noise(arrs):
        arr = np.array(arrs, dtype=np.float32)
        noise = np.random.normal(0, noise_level_small, arr.shape).astype(np.float32)
        return arr + noise
    
    base_samples = np.array(base_samples, dtype=np.float32)
    shifted = add_noise(shifted)
    rotated = add_noise(rotated)
    cropped = add_noise(cropped)
    stretched = add_noise(stretched)
    
    # Combine all augmented data
    combined_data = np.concatenate([base_samples, shifted, rotated, cropped, stretched], axis=0)
    
    # Create corresponding Freeman codes (replicated for augmentations)
    combined_freeman_codes = train_freeman_codes * 5  # 5 versions: original + 4 augmentations
    
    # Create corresponding labels
    label_ids = np.array([class_to_id[l] for l in train_labels_text])
    combined_labels = np.tile(label_ids, 5)
    
    # Create pairs for training with edit distance labels
    n_tra = len(combined_data)
    max_pairs = 50000  # Limit pairs for memory efficiency
    all_pairs = list(combinations(range(n_tra), 2))
    if len(all_pairs) > max_pairs:
        random.Random(random_seed).shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]
    
    combinations_list = np.array(all_pairs)
    x1 = np.array([combined_data[i] for i in combinations_list[:, 0]])
    x2 = np.array([combined_data[i] for i in combinations_list[:, 1]])
    train_x = [x1, x2]
    
    # MAIN CHANGE: Use edit distance instead of binary labels
    print("Computing edit distances for training pairs...")
    train_y = []
    for idx1, idx2 in tqdm(combinations_list, desc="Computing edit distances"):
        codes1 = combined_freeman_codes[idx1]
        codes2 = combined_freeman_codes[idx2]
        edit_dist = compute_combined_edit_distance(codes1, codes2)
        train_y.append(edit_dist)
    
    train_y = np.array(train_y, dtype=np.float32)
    
    print(f"Edit distance statistics:")
    print(f"  Min: {train_y.min():.4f}")
    print(f"  Max: {train_y.max():.4f}")
    print(f"  Mean: {train_y.mean():.4f}")
    print(f"  Std: {train_y.std():.4f}")
    
    # Show distribution
    plt.figure(figsize=(10, 6))
    plt.hist(train_y, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Edit Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Edit Distances in Training Pairs')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_x, train_y, base_samples, combined_labels, combined_data, combined_freeman_codes

# ===== STKNet Architecture (preserved from original) =====
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

# GlobalInfoProcess and LocalInfoProcess classes preserved exactly from original
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
        c1 = self.conv1(x); c2 = self.conv2(x); c3 = self.conv3(x); c4 = self.conv4(x); c5 = self.conv5(x)
        c6 = self.conv6(x); c7 = self.conv7(x); c8 = self.conv8(x); c9 = self.conv9(x); c10 = self.conv10(x); c11 = self.conv11(x)
        cat1 = self.flatten(tf.concat([
            self.global_avg_pooling(c1), self.global_avg_pooling(c2), self.global_avg_pooling(c3),
            self.global_avg_pooling(c4), self.global_avg_pooling(c5), self.global_avg_pooling(c6),
            self.global_avg_pooling(c7), self.global_avg_pooling(c8), self.global_avg_pooling(c9),
            self.global_avg_pooling(c10), self.global_avg_pooling(c11)], axis=-1))
        cat2 = self.flatten(tf.concat([
            self.global_max_pooling(c1), self.global_max_pooling(c2), self.global_max_pooling(c3),
            self.global_max_pooling(c4), self.global_max_pooling(c5), self.global_max_pooling(c6),
            self.global_max_pooling(c7), self.global_max_pooling(c8), self.global_max_pooling(c9),
            self.global_max_pooling(c10), self.global_max_pooling(c11)], axis=-1))
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

# MODIFIED: Loss function adapted for continuous targets
def continuous_similarity_loss(y_true, y_pred):
    """
    Loss function for continuous similarity targets.
    y_true: edit distances (0=identical, 1=completely different)
    y_pred: predicted similarities (0=different, 1=identical)
    
    We want: low edit distance -> high similarity
    """
    # Convert edit distance to similarity: similarity = 1 - edit_distance
    y_true_similarity = 1.0 - y_true
    
    # Use MSE between predicted similarity and target similarity
    mse_loss = K.mean(K.square(y_pred - y_true_similarity))
    
    return mse_loss

# Alternative loss that's closer to original STKNet
def margin_based_continuous_loss(y_true, y_pred, margin=0.3):
    """
    Modified margin loss for continuous targets.
    """
    # Convert edit distance to similarity
    y_true_similarity = 1.0 - y_true
    
    # For similar pairs (high similarity), we want high predictions
    # For dissimilar pairs (low similarity), we want low predictions
    loss_similar = K.maximum(0.0, margin + y_true - y_pred)**2 * y_true_similarity
    loss_dissimilar = y_pred**2 * (1.0 - y_true_similarity)
    
    return loss_similar + loss_dissimilar

# ===== KNN Prediction Function =====
def knn_vote(pred_matrix: np.ndarray, train_labels: np.ndarray, k: int):
    """KNN voting based on similarity scores."""
    if pred_matrix.size == 0:
        return np.array([], dtype=np.int32)
    # For similarity scores, higher is better
    top_k = np.argpartition(-pred_matrix, k, axis=1)[:, :k]  # Note the negative sign
    votes = train_labels[top_k]
    def row_mode(arr):
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=arr)
    return row_mode(votes)

def compute_character_level_accuracy(predicted_chars, true_chars):
    """
    Compute character-level accuracy by comparing reconstructed Freeman codes.
    
    Args:
        predicted_chars: List of predicted Freeman code sequences (one per character)
        true_chars: List of ground-truth Freeman code sequences (one per character)
        
    Returns:
        float: Accuracy between 0 and 1
    """
    correct = 0
    total = len(predicted_chars)
    
    for pred, true in zip(predicted_chars, true_chars):
        # Remove any stroke separators and compare
        pred_clean = pred.replace('-', '').strip()
        true_clean = true.replace('-', '').strip()
        if pred_clean == true_clean:
            correct += 1
            
    return correct / total if total > 0 else 0.0

def reconstruct_character_freeman(stroke_sequences):
    """
    Reconstruct character-level Freeman code from stroke sequences.
    
    Args:
        stroke_sequences: List of Freeman code sequences for each stroke
        
    Returns:
        str: Combined Freeman code for the character
    """
    # Join strokes with a separator that won't conflict with Freeman codes
    return '-'.join(''.join(codes) for codes in stroke_sequences)

def model_predict_hanzi_edit_distance(test_samples, test_freeman_codes, ref_samples, ref_freeman_codes, 
                                    ref_labels, model, k=None, test_char_labels=None, ref_char_labels=None):
    """
    Enhanced prediction function with character-level evaluation.
    
    Args:
        test_samples: Test trajectory samples
        test_freeman_codes: Freeman codes for test samples
        ref_samples: Reference trajectory samples
        ref_freeman_codes: Freeman codes for reference samples
        ref_labels: Labels for reference samples
        model: Trained STKNet model
        k: Number of neighbors for kNN
        test_char_labels: Optional list of character labels for test samples
        ref_char_labels: Optional list of character labels for reference samples
        
    Returns:
        tuple: (predicted_labels, character_accuracy, char_level_metrics)
    """
    if k is None:
        k = 5
        
    # Get pairwise distances using the model
    n_test = len(test_samples)
    n_ref = len(ref_samples)
    
    if n_test == 0 or n_ref == 0:
        print(f"Warning: Skipping prediction (n_test={n_test}, n_ref={n_ref}).")
        return np.array([], dtype=np.int32), np.zeros((n_test, n_ref), dtype=np.float32), {}
    
    # Initialize distance matrix
    dist_matrix = np.zeros((n_test, n_ref))
    
    # Compute distances in batches to save memory
    batch_size = 32
    for i in tqdm(range(0, n_test, batch_size), desc="Computing distances"):
        batch_test = np.array(test_samples[i:i+batch_size])
        batch_dist = []
        
        for j in range(0, n_ref, batch_size):
            batch_ref = np.array(ref_samples[j:j+batch_size])
            
            # Create pairs for the current batch
            x1 = np.repeat(batch_test, len(batch_ref), axis=0)
            x2 = np.tile(batch_ref, (len(batch_test), 1, 1))
            
            # Predict distances
            batch_pred = model.predict([x1, x2], verbose=0).flatten()
            batch_dist.append(batch_pred.reshape(len(batch_test), -1))
            
        # Stack distances for this test batch
        dist_matrix[i:i+len(batch_test)] = np.hstack(batch_dist)
    
    # Get k nearest neighbors for each test sample
    knn_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    
    # Predict labels using kNN voting
    predicted_labels = []
    for i in range(n_test):
        neighbor_labels = [ref_labels[idx] for idx in knn_indices[i]]
        predicted_labels.append(Counter(neighbor_labels).most_common(1)[0][0])
    
    # Character-level evaluation if character labels are provided
    char_level_metrics = {}
    if test_char_labels is not None and ref_char_labels is not None:
        # Reconstruct character-level Freeman codes
        test_chars = {}
        ref_chars = {}
        
        # Group strokes by character for test set
        char_strokes = {}
        for i, (codes, char_label) in enumerate(zip(test_freeman_codes, test_char_labels)):
            if char_label not in char_strokes:
                char_strokes[char_label] = []
            char_strokes[char_label].append(codes)
        
        # Reconstruct characters for test set
        test_char_codes = {}
        for char_label, strokes in char_strokes.items():
            test_char_codes[char_label] = reconstruct_character_freeman(strokes)
        
        # Do the same for reference set
        ref_char_strokes = {}
        for i, (codes, char_label) in enumerate(zip(ref_freeman_codes, ref_char_labels)):
            if char_label not in ref_char_strokes:
                ref_char_strokes[char_label] = []
            ref_char_strokes[char_label].append(codes)
        
        ref_char_codes = {}
        for char_label, strokes in ref_char_strokes.items():
            ref_char_codes[char_label] = reconstruct_character_freeman(strokes)
        
        # Compute character-level accuracy
        char_accuracy = compute_character_level_accuracy(
            list(test_char_codes.values()), 
            [ref_char_codes.get(lbl, '') for lbl in test_char_codes.keys()]
        )
        
        char_level_metrics = {
            'character_accuracy': char_accuracy,
            'num_test_chars': len(test_char_codes),
            'num_ref_chars': len(ref_char_codes)
        }
    
    return np.array(predicted_labels), dist_matrix, char_level_metrics
    
    # Create pairwise comparison arrays
    X1 = np.repeat(np.array(test_samples), n_ref, axis=0)
    X2 = np.tile(np.array(ref_samples), (n_test, 1, 1))
    
    print(f"Computing STKNet similarities for {n_test} test vs {n_ref} reference samples...")
    similarities = model.predict([X1, X2], batch_size=1024, verbose=1)
    similarities_2d = similarities.reshape(n_test, n_ref)
    
    # KNN voting based on STKNet similarities
    predicted_labels = knn_vote(similarities_2d, ref_labels, k=k)
    
    return predicted_labels, similarities_2d

# ===== Main Function =====
if __name__ == "__main__":
    print("=== STKNet with Edit Distance for Hanzi Recognition ===")
    
    # Load data
    print("Loading Hanzi data...")
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    df["strock_number"] = pd.to_numeric(df["strock_number"], errors="coerce").fillna(0).astype(int)
    
    # Writer-independent split to avoid overlap issues
    writers = sorted(df["writer"].unique())
    random.Random(random_seed).shuffle(writers)
    n_test_writers = max(1, int(len(writers) * 0.2))
    test_writers = set(writers[:n_test_writers])
    
    train_df = df[~df["writer"].isin(test_writers)].copy()
    test_df = df[df["writer"].isin(test_writers)].copy()
    
    print(f"Train writers: {len(writers) - n_test_writers}, Test writers: {n_test_writers}")
    
    # Build samples with Freeman codes and track character information
    train_samples_xy, train_labels_text, train_ids, train_freeman_codes = build_samples_from_df(train_df)
    test_samples_xy, test_labels_text, test_ids, test_freeman_codes = build_samples_from_df(test_df)
    
    # Extract character labels (assuming format is 'writer|char_nr' in train_ids/test_ids)
    train_char_labels = [f"{id_.split('|')[0]}_{id_.split('|')[1]}" for id_ in train_ids]
    test_char_labels = [f"{id_.split('|')[0]}_{id_.split('|')[1]}" for id_ in test_ids]
    
    print(f"Built {len(train_samples_xy)} train samples, {len(test_samples_xy)} test samples.")
    
    # Check class overlap
    train_classes = set(train_labels_text)
    test_classes = set(test_labels_text)
    overlap = train_classes & test_classes
    print(f"Unique classes — train: {len(train_classes)}, test: {len(test_classes)}, overlap: {len(overlap)}")
    

    if len(overlap) == 0:
        print("No class overlap detected, but edit distance approach can still work!")
        print("The model will learn similarity based on Freeman code patterns.")
    
    # Limit classes for development
    if limit_classes_to_top:
        cnt = Counter(train_labels_text)
        top_classes = set([c for c, _ in cnt.most_common(limit_classes_to_top)])
        keep_train = [i for i, l in enumerate(train_labels_text) if l in top_classes]

        # NO filter test classes - To see how well we generalize
        train_samples_xy = [train_samples_xy[i] for i in keep_train]
        train_labels_text = [train_labels_text[i] for i in keep_train]
        train_freeman_codes = [train_freeman_codes[i] for i in keep_train]
    
    # Encode labels - include ALL classes from both train and test
    all_classes = sorted(list(set(train_labels_text + test_labels_text)))
    class_to_id = {c: i for i, c in enumerate(all_classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    
    train_label_ids = np.array([class_to_id[c] for c in train_labels_text])
    test_label_ids = np.array([class_to_id[c] for c in test_labels_text])
    
    num_classes = len(all_classes)
    print(f"Working with {num_classes} total classes")
    
    # Create training data with edit distance labels
    print("Creating augmented training data with edit distance labels...")
    train_x, train_y, ref_samples, combined_labels, combined_data, combined_freeman_codes = createData_hanzi_edit_distance(
        train_samples_xy, train_labels_text, train_freeman_codes)
    
    print(f"Created {len(train_x[0])} training pairs with edit distance targets")
    
    # Build STKNet model (same architecture as original)
    print("Building STKNet model...")
    trajectory_length = desired_length
    num_dimensions = 2
    
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
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
    
    # MODIFIED: Use continuous similarity loss instead of binary classification loss
    model.compile(loss=continuous_similarity_loss, 
                 optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 metrics=['mae'])  # Mean Absolute Error is meaningful for continuous targets
    
    print("STKNet Model Summary:")
    model.summary()
    
    # Train STKNet model
    print("Training STKNet model with edit distance targets...")
    epochs = 2  # Same as original
    batch_size = 1024
    
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, 
                       shuffle=True, verbose=1, validation_split=0.1)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("STKNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title("STKNet Training MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Show some sample predictions vs targets
    sample_indices = np.random.choice(len(train_y), min(1000, len(train_y)), replace=False)
    sample_pred = model.predict([train_x[0][sample_indices], train_x[1][sample_indices]], verbose=0)
    sample_true = train_y[sample_indices]
    plt.scatter(sample_true, sample_pred.flatten(), alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('True Similarity (1 - Edit Distance)')
    plt.ylabel('Predicted Similarity')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create LSTM baseline for comparison
    print("Training LSTM baseline...")
    
    def create_lstm_model(input_shape, num_classes):
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    model_lstm = create_lstm_model((trajectory_length, num_dimensions), num_classes)
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train LSTM on combined augmented data
    model_lstm.fit(combined_data, combined_labels, batch_size=1024, epochs=50, verbose=1)
    
    # Prepare reference data for KNN
    ref_samples_2d = [s[:, :2] for s in train_samples_xy]
    ref_labels = train_label_ids
    ref_freeman_codes_original = train_freeman_codes
    
    # Prepare test data
    test_samples_2d = [s[:, :2] for s in test_samples_xy]
    
    # Evaluation
    if len(test_samples_2d) == 0 or len(ref_samples_2d) == 0:
        print("Cannot evaluate: empty test or reference set.")
    else:
        # STKNet + KNN evaluation
        print("\nEvaluating STKNet + KNN with character-level metrics...")
        predicted_labels, similarity_matrix, char_metrics = model_predict_hanzi_edit_distance(
            test_samples_xy, test_freeman_codes, 
            ref_samples, ref_freeman_codes, 
            combined_labels, model, k=5,
            test_char_labels=test_char_labels,
            ref_char_labels=[f"{id_.split('|')[0]}_{id_.split('|')[1]}" for id_ in [x.split('|') for x in train_ids]]
        )
        
        if test_label_ids.size > 0 and predicted_labels.size > 0:
            accuracy_stknet = np.mean(predicted_labels == test_label_ids)
            print(f"STKNet + KNN accuracy: {accuracy_stknet:.4f}")
        else:
            accuracy_stknet = float('nan')
            print("STKNet + KNN accuracy: N/A (no valid predictions)")
        
        # LSTM baseline prediction
        print("Evaluating LSTM baseline...")
        test_samples_array = np.array(test_samples_2d)
        lstm_predictions = model_lstm.predict(test_samples_array, batch_size=1024, verbose=1)
        predicted_labels_lstm = np.argmax(lstm_predictions, axis=1)
        accuracy_lstm = np.mean(predicted_labels_lstm == test_label_ids)
        print(f"LSTM baseline accuracy: {accuracy_lstm:.4f}")
        
        # Additional analysis: Edit distance correlation
        print("\nAnalyzing edit distance correlations...")
        if len(test_samples_2d) > 0 and len(ref_samples_2d) > 0:
            # Sample some test-reference pairs to analyze
            n_sample_pairs = min(1000, len(test_samples_2d) * len(ref_samples_2d))
            sample_test_idx = np.random.choice(len(test_samples_2d), 
                                             min(100, len(test_samples_2d)), replace=False)
            sample_ref_idx = np.random.choice(len(ref_samples_2d), 
                                            min(100, len(ref_samples_2d)), replace=False)
            
            edit_distances = []
            stknet_similarities = []
            
            for t_idx in sample_test_idx[:10]:  # Limit for computational efficiency
                for r_idx in sample_ref_idx[:10]:
                    edit_dist = compute_combined_edit_distance(
                        test_freeman_codes[t_idx], ref_freeman_codes_original[r_idx])
                    stknet_sim = model.predict([
                        np.array([test_samples_2d[t_idx]]), 
                        np.array([ref_samples_2d[r_idx]])
                    ], verbose=0)[0, 0]
                    
                    edit_distances.append(edit_dist)
                    stknet_similarities.append(stknet_sim)
            
            if len(edit_distances) > 0:
                correlation = np.corrcoef(edit_distances, stknet_similarities)[0, 1]
                print(f"Correlation between edit distance and STKNet similarity: {correlation:.4f}")
                
                plt.figure(figsize=(10, 6))
                plt.scatter(edit_distances, stknet_similarities, alpha=0.6)
                plt.xlabel('Edit Distance (Freeman Codes)')
                plt.ylabel('STKNet Predicted Similarity')
                plt.title(f'Edit Distance vs STKNet Similarity (r={correlation:.3f})')
                plt.grid(True)
                plt.show()
    
    # Per-class performance analysis
    print("\nPer-class performance analysis:")
    for class_id in range(min(10, num_classes)):  # Show first 10 classes
        class_mask = test_label_ids == class_id
        if np.sum(class_mask) > 0 and predicted_labels_stknet.size > 0:
            class_acc = np.mean(predicted_labels_stknet[class_mask] == class_id)
            class_name = id_to_class[class_id]
            print(f"Class {class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")
    
    # Save results
    print(f"\nSaving results to {save_results_root}...")
    np.save(os.path.join(save_results_root, "all_classes.npy"), np.array(all_classes))
    np.save(os.path.join(save_results_root, "test_labels.npy"), test_label_ids)
    if 'predicted_labels_stknet' in locals():
        np.save(os.path.join(save_results_root, "predicted_stknet_edit.npy"), predicted_labels_stknet)
    if 'predicted_labels_lstm' in locals():
        np.save(os.path.join(save_results_root, "predicted_lstm.npy"), predicted_labels_lstm)
    if 'similarity_matrix' in locals():
        np.save(os.path.join(save_results_root, "similarity_matrix_edit.npy"), similarity_matrix)
    
    # Save models
    model.save(os.path.join(save_results_root, "stknet_hanzi_edit_model.keras"))
    model_lstm.save(os.path.join(save_results_root, "lstm_baseline_model.keras"))
    
    print("Training and evaluation completed!")
    print(f"Results saved in: {save_results_root}")
    
    # Final summary
    print(f"\n=== FINAL RESULTS (Edit Distance Approach) ===")
    print(f"Dataset: {num_classes} Hanzi classes")
    print(f"Train samples: {len(train_samples_xy)} (from {len(writers) - n_test_writers} writers)")
    print(f"Test samples: {len(test_samples_xy)} (from {n_test_writers} writers)")
    print(f"Training pairs: {len(train_x[0])} with edit distance targets")
    if 'accuracy_stknet' in locals():
        print(f"STKNet + KNN accuracy: {accuracy_stknet:.4f}")
    if 'accuracy_lstm' in locals():
        print(f"LSTM baseline accuracy: {accuracy_lstm:.4f}")
    
    # Print character-level metrics if available
    if 'char_metrics' in locals() and char_metrics:
        print("\n=== CHARACTER-LEVEL METRICS ===")
        print(f"Test characters: {char_metrics.get('num_test_chars', 0)}")
        print(f"Reference characters: {char_metrics.get('num_ref_chars', 0)}")
        print(f"Character-level accuracy: {char_metrics.get('character_accuracy', 0):.4f}")
    