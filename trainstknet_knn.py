import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras.models import Model
import random

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("STKNET + KNN LIBRARY LOOKUP (NO DECODER)")
print("="*60)

# ===== Freeman Chain Code Utilities =====
FREEMAN_8 = {
    0: (1, 0),   1: (1, 1),   2: (0, 1),   3: (-1, 1),
    4: (-1, 0),  5: (-1,-1),  6: (0,-1),  7: (1,-1),
    8: (0, 0)    # Hyphen / Hold
}

def parse_freeman_string_with_hyphen(s: str) -> List[int]:
    """
    Parse Freeman code string, extracting digit sequences AND hyphens.
    Hyphen '-' is mapped to 8.
    """
    s = str(s)
    codes = []
    # Iterate chars to preserve order
    # Filter for digits 0-7 and '-'
    for char in s:
        if char == '-':
            codes.append(8)
        elif char.isdigit() and int(char) <= 7:
            codes.append(int(char))
            
    return codes

def freeman_to_xy(codes: List[int], start=(0.0, 0.0)) -> np.ndarray:
    """Convert Freeman codes to cumulative 2D coordinates with pen info."""
    x, y = start
    pts = [(x, y)]
    is_hyphen = []
    
    # Store hyphen status for start point? Assume 0
    is_hyphen.append(0.0)
    
    for c in codes:
        dx, dy = FREEMAN_8.get(c, (0, 0))
        x += dx; y += dy
        pts.append((x, y))
        
        # If code was 8, this step was a hyphen
        is_hyphen.append(1.0 if c == 8 else 0.0)
        
    xy = np.array(pts, dtype=np.float32)
    pen = np.ones((xy.shape[0], 1), dtype=np.float32)
    hyphen = np.array(is_hyphen, dtype=np.float32).reshape(-1, 1)
    
    # Return XY + Pen + Hyphen
    return np.concatenate([xy, pen, hyphen], axis=1)

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
        
    # Combine back with other channels (pen, hyphen)
    others = traj[:, 2:] 
    return np.concatenate([xy_norm, others], axis=1).astype(np.float32)

def resample_xy(traj: np.ndarray, target_len: int) -> np.ndarray:
    """Resample trajectory to fixed length."""
    if len(traj) == 0:
        ch = traj.shape[1]
        return np.zeros((target_len, ch), dtype=np.float32)
    n = traj.shape[0]
    if n == target_len:
        return traj.astype(np.float32)
    
    old_idx = np.arange(n)
    new_idx = np.linspace(0, n - 1, target_len)
    
    result = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    
    # XY: Linear
    result[:, 0] = interp1d(old_idx, traj[:, 0], kind='linear', fill_value="extrapolate")(new_idx)
    result[:, 1] = interp1d(old_idx, traj[:, 1], kind='linear', fill_value="extrapolate")(new_idx)
    
    # Others (Pen, Hyphen): Nearest
    # We want preserve binary nature of flags
    for i in range(2, traj.shape[1]):
        result[:, i] = interp1d(old_idx, traj[:, i], kind='nearest', fill_value="extrapolate")(new_idx)
        
    return result

# ===== DATA LOADING =====
def build_stroke_samples_from_df(df):
    stroke_trajectories = []
    stroke_labels = []
    stroke_ids = []
    stroke_writers = []
    stroke_char_nrs = []
    stroke_hanzis = []
    skipped = 0
    
    stroke_counter = 0
    char_to_strokes = {}  # Maps (writer, char_nr) -> list of stroke indices
    
    for (writer, char_nr), g in df.groupby(['writer', 'character_nr']):
        g = g.sort_values('strock_number')
        current_pos = (0.0, 0.0)
        hanzi = str(g['hanzi'].iloc[0])
        char_key = (writer, char_nr)
        char_to_strokes[char_key] = []
        
        for _, row in g.iterrows():
            freeman_str = str(row['strock_label_freeman'])
            codes = parse_freeman_string_with_hyphen(freeman_str)
            if not codes:
                skipped += 1
                continue
            
            traj = freeman_to_xy(codes, start=current_pos)
            if len(traj) < 2:
                skipped += 1
                continue
            
            traj = normalize_xy(traj)
            traj = resample_xy(traj, 64)
            
            traj_6d = np.zeros((64, 6), dtype=np.float32)
            traj_6d[:, :4] = traj
            
            dx = np.zeros(64)
            dy = np.zeros(64)
            dx[1:] = traj[1:, 0] - traj[:-1, 0]
            dy[1:] = traj[1:, 1] - traj[:-1, 1]
            dx[0] = dx[1]
            dy[0] = dy[1]
            
            traj_6d[:, 4] = dx
            traj_6d[:, 5] = dy
            
            stroke_trajectories.append(traj_6d)
            stroke_labels.append(freeman_str)
            stroke_ids.append(stroke_counter)
            stroke_writers.append(writer)
            stroke_char_nrs.append(char_nr)
            stroke_hanzis.append(hanzi)
            char_to_strokes[char_key].append(stroke_counter)
            stroke_counter += 1
            
            current_pos = (traj[-1, 0], traj[-1, 1])
    
    print(f"Built {len(stroke_trajectories)} stroke samples, skipped {skipped}")
    print(f"Total characters: {len(char_to_strokes)}")
    return (stroke_trajectories, stroke_labels, stroke_ids, 
            stroke_writers, stroke_char_nrs, stroke_hanzis, char_to_strokes)


#===== CHARACTER-LEVEL EVALUATION =====
def evaluate_character_level(char_to_strokes, stroke_true_labels, stroke_pred_labels, 
                             stroke_writers, stroke_hanzis, freeman_to_hanzi_map, 
                             all_stroke_embeddings=None, canonical_embeddings=None, 
                             canonical_hanzis_list=None, scenario_name=""):

    char_correct = 0
    char_total = 0
    samples = []
    
    # Build KNN for character-level matching if embeddings provided
    char_knn = None
    if canonical_embeddings is not None and len(canonical_embeddings) > 0:
        char_knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        char_knn.fit(canonical_embeddings)
    
    for (writer, char_nr), stroke_indices in char_to_strokes.items():
        # Get true and predicted Freeman codes for all strokes
        true_codes = [stroke_true_labels[i] for i in stroke_indices if i < len(stroke_true_labels)]
        pred_codes = [stroke_pred_labels[i] for i in stroke_indices if i < len(stroke_pred_labels)]
        
        if len(true_codes) != len(pred_codes) or len(true_codes) == 0:
            continue
            
        # Assemble character (concatenate all stroke codes)
        true_freeman = ''.join(true_codes)
        pred_freeman = ''.join(pred_codes)
        
        # Get Hanzi labels
        true_hanzi = stroke_hanzis[stroke_indices[0]] if stroke_indices else "?"
        


        pred_hanzi = freeman_to_hanzi_map.get(pred_freeman, None)
        
        # Embedding-based fallback
        if pred_hanzi is None and char_knn is not None and all_stroke_embeddings is not None:
            # Compute average embedding for predicted character
            pred_embeddings = []
            for idx in stroke_indices:
                if idx < len(all_stroke_embeddings):
                    pred_embeddings.append(all_stroke_embeddings[idx])
            
            if pred_embeddings:
                pred_char_embedding = np.mean(pred_embeddings, axis=0).reshape(1, -1)
                # Find nearest canonical character
                distances, indices = char_knn.kneighbors(pred_char_embedding)
                pred_hanzi = canonical_hanzis_list[indices[0][0]]
        
        # Final fallback
        if pred_hanzi is None:
            pred_hanzi = "未知"
        
        # FIXED: Compare Hanzi characters, not Freeman codes!
        # Same character can have different stroke sequences (order, count variations)
        match = (true_hanzi == pred_hanzi)  # Character-level match
        freeman_match = (true_freeman == pred_freeman)  # Stroke-level match
        
        if match:
            char_correct += 1
        char_total += 1
        
        # Store sample with both true and predicted Hanzi
        if len(samples) < 20:
            samples.append((true_hanzi, pred_hanzi, match, freeman_match))
    
    accuracy = (char_correct / char_total * 100) if char_total > 0 else 0.0
    
    print(f"CHARACTER-LEVEL RESULTS ({scenario_name}):")
    print(f"Character Recognition Accuracy: {accuracy:.2f}%")
    print(f"Correct: {char_correct}/{char_total}")

    
    # Sample predictions with true vs predicted
    print(f"\nSample Character Predictions:")
    print(f"{'True':<13} | {'Predicted':<14} | {'Result':<10}")
    print("-" * 45)
    for true_hanzi, pred_hanzi, match, freeman_match in samples:
        char_result = "Right" if match else "Wrong"
        freeman_status = "Right" if freeman_match else "Wrong"
        print(f"{true_hanzi:<12} | {pred_hanzi:<15} | {char_result:<12} (Freeman: {freeman_status})")
    
    return accuracy



# ===== WRITER-AWARE AUGMENTATION =====
def augment_stroke_writer_aware(trajectory, num_augmentations=2):
    """
    Augment stroke trajectories with transformations that simulate writer variations.
    Args:
        trajectory: (N, 6) array with [x, y, pen, hyphen, dx, dy]
        num_augmentations: number of augmented versions to create
    Returns:
        list of augmented trajectories
    """
    augmented = []
    
    for _ in range(num_augmentations):
        aug_traj = trajectory.copy()
        
        # Extract XY coordinates (first 2 channels)
        xy = aug_traj[:, :2].copy()
        
        # PHASE 1: Stronger augmentation for better writer variation robustness
        # 1. Random scaling (0.8-1.2) - simulates different hand sizes [INCREASED from 0.85-1.15]
        scale = np.random.uniform(0.8, 1.2)
        xy = xy * scale
        
        # 2. Random rotation (±30 degrees) - simulates writing angle variation [INCREASED from ±15]
        angle = np.random.uniform(-30, 30) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        xy = xy @ rotation_matrix.T
        
        # 3. Random shear (±0.2) - simulates writing slant variation
        shear_x = np.random.uniform(-0.2, 0.2)
        shear_matrix = np.array([[1, shear_x], [0, 1]])
        xy = xy @ shear_matrix.T
        
        # 4. Stronger random noise - simulates hand tremor/natural variation [INCREASED from 0.02]
        noise = np.random.normal(0, 0.03, xy.shape)
        xy = xy + noise
        
        # Update augmented trajectory with new XY
        aug_traj[:, :2] = xy
        
        # Re-normalize to maintain unit scale
        aug_traj = normalize_xy(aug_traj)
        
        augmented.append(aug_traj)
    
    return augmented


# ===== TRIPLET LOSS TRAINING =====
    return loss


# ===== CONTRASTIVE LOSS WITH HARD NEGATIVE MINING =====
def contrastive_loss(margin=1.0):

    def loss(y_true, y_pred):
        # y_true: (batch, 1) - labels where 1=positive pair, 0=negative pair
        # y_pred: (batch, 2*embedding_dim) - concatenated pairs [emb1, emb2]
        
        embedding_dim = tf.shape(y_pred)[1] // 2
        
        # Split into two embeddings
        emb1 = y_pred[:, :embedding_dim]
        emb2 = y_pred[:, embedding_dim:]
        
        # Compute Euclidean distance
        distance = tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis=1) + 1e-8)
        
        # y_true = 1 for positive pairs (same class), 0 for negative pairs
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, [-1])
        
        # Positive loss: minimize distance
        positive_loss = y_true * tf.square(distance)
        
        # Negative loss: maximize distance (push apart by margin)
        negative_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - distance, 0.0))
        
        # Combine losses
        loss_value = tf.reduce_mean(positive_loss + negative_loss)
        
        return loss_value
    
    return loss



# ===== HARD NEGATIVE MINING =====
def mine_hard_negatives_batch(anchor_emb, positive_emb, all_emb, all_labels, anchor_labels, 
                               strategy='semi-hard', margin=1.0):

    import tensorflow as tf
    
    # Convert to TF tensors if needed
    if not isinstance(anchor_emb, tf.Tensor):
        anchor_emb = tf.constant(anchor_emb, dtype=tf.float32)
    if not isinstance(positive_emb, tf.Tensor):
        positive_emb = tf.constant(positive_emb, dtype=tf.float32)
    if not isinstance(all_emb, tf.Tensor):
        all_emb = tf.constant(all_emb, dtype=tf.float32)
    
    N = tf.shape(anchor_emb)[0]
    M = tf.shape(all_emb)[0]
    
    # Compute anchor-positive distances
    pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=1)  # (N,)
    
    # Compute distances from anchors to all embeddings
    # anchor_emb: (N, D), all_emb: (M, D)
    # distances: (N, M)
    anchor_expanded = tf.expand_dims(anchor_emb, 1)  # (N, 1, D)
    all_expanded = tf.expand_dims(all_emb, 0)  # (1, M, D)
    distances = tf.reduce_sum(tf.square(anchor_expanded - all_expanded), axis=2)  # (N, M)
    
    # Create mask for valid negatives (different label from anchor)
    anchor_labels_expanded = tf.expand_dims(anchor_labels, 1)  # (N, 1)
    all_labels_expanded = tf.expand_dims(all_labels, 0)  # (1, M)
    valid_negatives = tf.not_equal(anchor_labels_expanded, all_labels_expanded)  # (N, M)
    
    # Apply large distance to invalid negatives
    large_dist = tf.constant(1e10, dtype=tf.float32)
    masked_distances = tf.where(valid_negatives, distances, large_dist)
    
    if strategy == 'hard':
        # Select closest negative (hardest)
        negative_indices = tf.argmin(masked_distances, axis=1)
    
    elif strategy == 'semi-hard':
        # Select negatives where: d(a,p) < d(a,n) < d(a,p) + margin
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)  # (N, 1)
        
        # Semi-hard condition: pos_dist < neg_dist < pos_dist + margin
        semi_hard_mask = tf.logical_and(
            distances > pos_dist_expanded,
            distances < pos_dist_expanded + margin
        )
        semi_hard_mask = tf.logical_and(semi_hard_mask, valid_negatives)
        
        # If no semi-hard negatives, fall back to hard negatives
        semi_hard_distances = tf.where(semi_hard_mask, distances, large_dist)
        has_semi_hard = tf.reduce_any(semi_hard_distances < large_dist, axis=1)
        
        # Choose closest semi-hard, or closest hard if no semi-hard exists
        semi_hard_indices = tf.argmin(semi_hard_distances, axis=1)
        hard_indices = tf.argmin(masked_distances, axis=1)
        negative_indices = tf.where(has_semi_hard, semi_hard_indices, hard_indices)
    
    else:  # mixed
        # 50% hard, 50% semi-hard
        # Use random selection per sample
        use_hard = tf.random.uniform((N,), 0, 1) < 0.5
        
        # Hard negatives
        hard_indices = tf.argmin(masked_distances, axis=1)
        
        # Semi-hard negatives
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)
        semi_hard_mask = tf.logical_and(
            tf.logical_and(distances > pos_dist_expanded, distances < pos_dist_expanded + margin),
            valid_negatives
        )
        semi_hard_distances = tf.where(semi_hard_mask, distances, large_dist)
        has_semi_hard = tf.reduce_any(semi_hard_distances < large_dist, axis=1)
        semi_hard_indices = tf.argmin(semi_hard_distances, axis=1)
        
        # Fall back to hard if no semi-hard
        semi_hard_final = tf.where(has_semi_hard, semi_hard_indices, hard_indices)
        
        negative_indices = tf.where(use_hard, hard_indices, semi_hard_final)
    
    return negative_indices.numpy()


def triplet_loss_with_online_mining(margin=1.0, mining_strategy='semi-hard'):

    def loss(y_true, y_pred):
        # y_true: (batch, 1) contains Freeman code labels
        # y_pred: (batch, embedding_dim) contains embeddings
        
        batch_size = tf.shape(y_pred)[0]
        embedding_dim = tf.shape(y_pred)[1]
        
        # Assume batch is structured as [anchor, positive, ...] pairs
        # For simplicity, we'll use first half as anchors, second half as positives
        # and mine negatives from the entire batch
        
        half_batch = batch_size // 2
        anchor_emb = y_pred[:half_batch]
        positive_emb = y_pred[half_batch:half_batch*2]
        
        anchor_labels = y_true[:half_batch]
        all_labels = y_true
        
        # Compute anchor-positive distance
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=1)
        
        # Mine hard negatives
        # Compute all pairwise distances
        anchor_expanded = tf.expand_dims(anchor_emb, 1)  # (N, 1, D)
        all_expanded = tf.expand_dims(y_pred, 0)  # (1, M, D)
        distances = tf.reduce_sum(tf.square(anchor_expanded - all_expanded), axis=2)  # (N, M)
        
        # Mask out same-label pairs
        anchor_labels_exp = tf.expand_dims(anchor_labels, 1)
        all_labels_exp = tf.expand_dims(all_labels, 0)
        valid_negatives = tf.cast(tf.not_equal(anchor_labels_exp, all_labels_exp), tf.float32)
        
        # Apply large distance to invalid negatives
        masked_distances = distances + (1.0 - valid_negatives) * 1e10
        
        if mining_strategy == 'hard':
            # Select minimum distance negative
            neg_dist = tf.reduce_min(masked_distances, axis=1)
        elif mining_strategy == 'semi-hard':
            # Select semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
            pos_dist_exp = tf.expand_dims(pos_dist, 1)
            semi_hard_mask = tf.logical_and(
                distances > pos_dist_exp,
                distances < pos_dist_exp + margin
            )
            semi_hard_mask = tf.logical_and(semi_hard_mask, tf.cast(valid_negatives, tf.bool))
            
            semi_hard_distances = tf.where(semi_hard_mask, distances, 1e10)
            neg_dist = tf.reduce_min(semi_hard_distances, axis=1)
            
            # Fall back to hard if no semi-hard found
            has_semi_hard = tf.reduce_min(semi_hard_distances, axis=1) < 1e10
            hard_dist = tf.reduce_min(masked_distances, axis=1)
            neg_dist = tf.where(has_semi_hard, neg_dist, hard_dist)
        else:  # mixed
            # 50-50 mix
            hard_dist = tf.reduce_min(masked_distances, axis=1)
            
            pos_dist_exp = tf.expand_dims(pos_dist, 1)
            semi_hard_mask = tf.logical_and(
                tf.logical_and(distances > pos_dist_exp, distances < pos_dist_exp + margin),
                tf.cast(valid_negatives, tf.bool)
            )
            semi_hard_distances = tf.where(semi_hard_mask, distances, 1e10)
            semi_hard_dist = tf.reduce_min(semi_hard_distances, axis=1)
            
            use_semi = tf.random.uniform((half_batch,), 0, 1) < 0.5
            neg_dist = tf.where(use_semi, semi_hard_dist, hard_dist)
        
        # Compute triplet loss
        basic_loss = pos_dist - neg_dist + margin
        loss_value = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        
        return loss_value
    
    return loss


# Custom Keras Callback for Hard Negative Mining Statistics
class HardNegativeMiningCallback(keras.callbacks.Callback):

    def __init__(self, log_frequency=1):
        super().__init__()
        self.log_frequency = log_frequency
        self.epoch_losses = []
        self.mining_stats = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and (epoch + 1) % self.log_frequency == 0:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            self.epoch_losses.append(loss)
            
            print(f"\n[Epoch {epoch + 1}] Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if len(self.epoch_losses) >= 2:
                improvement = self.epoch_losses[-2] - self.epoch_losses[-1]
                print(f"  Loss improvement: {improvement:.4f}")





def create_contrastive_pairs_generator(X, y_codes, writers, encoder, batch_size=64, 
                                       use_augmentation=True, mining_strategy='semi-hard',
                                       margin=1.0, memory_bank_size=5000):

    # Group by Freeman code
    code_to_indices = {}
    for i, (code, writer) in enumerate(zip(y_codes, writers)):
        if code not in code_to_indices:
            code_to_indices[code] = []
        code_to_indices[code].append(i)
    
    # Filter codes with at least 2 samples for positive pairs
    valid_codes = [code for code, indices in code_to_indices.items() if len(indices) >= 2]
    all_codes = list(code_to_indices.keys())
    
    if len(valid_codes) < 2:
        raise ValueError("Not enough variety for pair generation")
    
    # Initialize memory bank
    memory_bank = {
        'embeddings': [],
        'labels': [],
        'indices': []
    }
    
    while True:  # Infinite generator
        input1_list = []
        input2_list = []
        labels_list = []
        indices1_list = []
        
        # Generate half positive pairs, half negative pairs
        num_positive = batch_size // 2
        num_negative = batch_size - num_positive
        
        # === POSITIVE PAIRS (same Freeman code) ===
        for _ in range(num_positive):
            code = np.random.choice(valid_codes)
            indices = code_to_indices[code]
            
            if len(indices) < 2:
                continue
            
            idx1, idx2 = np.random.choice(indices, size=2, replace=False)
            
            # Get data
            data1 = X[idx1]
            data2 = X[idx2]
            
            # Apply augmentation (70% probability - PHASE 1 enhancement)
            if use_augmentation and np.random.random() > 0.3:
                aug1 = augment_stroke_writer_aware(data1, num_augmentations=1)
                sample1 = aug1[0] if aug1 else data1
                
                aug2 = augment_stroke_writer_aware(data2, num_augmentations=1)
                sample2 = aug2[0] if aug2 else data2
            else:
                sample1 = data1
                sample2 = data2
            
            input1_list.append(sample1)
            input2_list.append(sample2)
            labels_list.append(1)  # positive pair
            indices1_list.append(idx1)
        
        if len(input1_list) > 0:
            input1_array = np.array(input1_list)
            emb1 = encoder.predict(input1_array, verbose=0)
            
            # Build candidate pool for negative mining
            if memory_bank['embeddings']:
                all_candidate_emb = np.vstack([
                    memory_bank['embeddings'][-memory_bank_size:],
                    emb1
                ])
                all_candidate_labels = np.concatenate([
                    memory_bank['labels'][-memory_bank_size:],
                    [y_codes[indices1_list[i]] for i in range(len(indices1_list))]
                ])
                all_candidate_indices = (
                    memory_bank['indices'][-memory_bank_size:] +
                    indices1_list
                )
            else:
                # Not enough data yet, use random negatives
                all_candidate_emb = emb1
                all_candidate_labels = np.array([y_codes[indices1_list[i]] for i in range(len(indices1_list))])
                all_candidate_indices = indices1_list
            
            # Mine hard negatives for new negative pairs
            for _ in range(num_negative):
                # Select random anchor
                anchor_idx = np.random.choice(len(X))
                anchor_code = y_codes[anchor_idx]
                anchor_data = X[anchor_idx]
                
                # Apply augmentation to anchor (70% probability)
                if use_augmentation and np.random.random() > 0.3:
                    aug_anchor = augment_stroke_writer_aware(anchor_data, num_augmentations=1)
                    anchor_sample = aug_anchor[0] if aug_anchor else anchor_data
                else:
                    anchor_sample = anchor_data
                
                # Compute anchor embedding
                anchor_emb = encoder.predict(np.array([anchor_sample]), verbose=0)[0]
                
                # Find hard negative (closest embedding with different label)
                distances = np.sum((all_candidate_emb - anchor_emb) ** 2, axis=1)
                
                # Mask same-label candidates
                valid_mask = np.array([all_candidate_labels[i] != anchor_code 
                                      for i in range(len(all_candidate_labels))])
                
                if not np.any(valid_mask):
                    # Fallback: random negative
                    neg_codes = [c for c in all_codes if c != anchor_code]
                    if neg_codes:
                        neg_code = np.random.choice(neg_codes)
                        neg_idx = np.random.choice(code_to_indices[neg_code])
                    else:
                        continue
                else:
                    # Apply mining strategy
                    masked_distances = distances.copy()
                    masked_distances[~valid_mask] = 1e10
                    
                    if mining_strategy == 'hard':
                        # Closest negative
                        neg_candidate_idx = np.argmin(masked_distances)
                    elif mining_strategy == 'semi-hard':
                        # Semi-hard: negatives within a distance range
                        # Find negatives closer than margin but not too close
                        semi_hard_mask = valid_mask & (distances < margin) & (distances > 0.1)
                        if np.any(semi_hard_mask):
                            semi_hard_distances = distances.copy()
                            semi_hard_distances[~semi_hard_mask] = 1e10
                            neg_candidate_idx = np.argmin(semi_hard_distances)
                        else:
                            # Fallback to hard
                            neg_candidate_idx = np.argmin(masked_distances)
                    else:  # mixed
                        if np.random.random() < 0.5:
                            neg_candidate_idx = np.argmin(masked_distances)
                        else:
                            semi_hard_mask = valid_mask & (distances < margin) & (distances > 0.1)
                            if np.any(semi_hard_mask):
                                semi_hard_distances = distances.copy()
                                semi_hard_distances[~semi_hard_mask] = 1e10
                                neg_candidate_idx = np.argmin(semi_hard_distances)
                            else:
                                neg_candidate_idx = np.argmin(masked_distances)
                    
                    neg_idx = all_candidate_indices[neg_candidate_idx]
                
                neg_data = X[neg_idx]
                
                input1_list.append(anchor_sample)
                input2_list.append(neg_data)
                labels_list.append(0)  # negative pair
            
            # Update memory bank with new embeddings
            memory_bank['embeddings'].extend(emb1.tolist())
            memory_bank['labels'].extend([y_codes[indices1_list[i]] for i in range(len(indices1_list))])
            memory_bank['indices'].extend(indices1_list)
            
            # Trim memory bank
            if len(memory_bank['embeddings']) > memory_bank_size:
                memory_bank['embeddings'] = memory_bank['embeddings'][-memory_bank_size:]
                memory_bank['labels'] = memory_bank['labels'][-memory_bank_size:]
                memory_bank['indices'] = memory_bank['indices'][-memory_bank_size:]
        
        if len(input1_list) == 0:
            continue
        
        # Convert to arrays
        input1_array = np.array(input1_list)
        input2_array = np.array(input2_list)
        labels_array = np.array(labels_list).reshape(-1, 1)
        
        # Shuffle pairs
        shuffle_idx = np.random.permutation(len(input1_array))
        input1_array = input1_array[shuffle_idx]
        input2_array = input2_array[shuffle_idx]
        labels_array = labels_array[shuffle_idx]
        
        yield [input1_array, input2_array], labels_array


def create_stroke_triplets(X, y_codes, writers, max_triplets=50000, use_augmentation=True, aug_factor=2):

    # Group strokes by Freeman code AND writer
    code_to_indices = {}
    code_writer_to_indices = {}  # (code, writer) -> indices
    
    for i, (code, writer) in enumerate(zip(y_codes, writers)):
        # By code only
        if code not in code_to_indices:
            code_to_indices[code] = []
        code_to_indices[code].append(i)
        
        # By code + writer
        key = (code, writer)
        if key not in code_writer_to_indices:
            code_writer_to_indices[key] = []
        code_writer_to_indices[key].append(i)
    
    # Filter codes with samples from multiple writers (for cross-writer positives)
    cross_writer_codes = []
    for code, indices in code_to_indices.items():
        code_writers = set(writers[i] for i in indices)
        if len(code_writers) >= 2:  # At least 2 different writers
            cross_writer_codes.append(code)
    
    # Also keep codes with at least 2 samples (fallback)
    same_writer_codes = [code for code, indices in code_to_indices.items() 
                        if len(indices) >= 2 and code not in cross_writer_codes]
    
    if len(cross_writer_codes) < 2 and len(same_writer_codes) < 2:
        print("⚠ Warning: Not enough variety for triplet generation")
        return np.array([]), np.array([])
    
    triplets = []
    cross_writer_count = 0
    
    # Generate triplets with cross-writer preference
    for _ in range(max_triplets):
        # Prefer cross-writer codes (70% of time)
        if cross_writer_codes and (len(same_writer_codes) == 0 or np.random.random() < 0.7):
            anchor_code = np.random.choice(cross_writer_codes)
            use_cross_writer = True
        else:
            anchor_code = np.random.choice(same_writer_codes) if same_writer_codes else np.random.choice(cross_writer_codes)
            use_cross_writer = False
        
        anchor_indices = code_to_indices[anchor_code]
        anchor_idx = np.random.choice(anchor_indices)
        anchor_writer = writers[anchor_idx]
        
        # Select positive: PREFER different writer with same code
        if use_cross_writer:
            # Cross-writer positive: different writer, same code
            diff_writer_indices = [i for i in anchor_indices if writers[i] != anchor_writer]
            if diff_writer_indices:
                positive_idx = np.random.choice(diff_writer_indices)
                cross_writer_count += 1
            else:
                # Fallback: same writer
                same_writer_indices = [i for i in anchor_indices if i != anchor_idx]
                if same_writer_indices:
                    positive_idx = np.random.choice(same_writer_indices)
                else:
                    continue
        else:
            # Same writer positive (traditional)
            same_writer_indices = [i for i in anchor_indices if i != anchor_idx]
            if same_writer_indices:
                positive_idx = np.random.choice(same_writer_indices)
            else:
                continue
        
        # Select SEMI-HARD NEGATIVE: different code, but prefer:
        # 1. Same writer as anchor (hard within-writer negative) 40% of time
        # 2. Common/confusable codes (semi-hard) 60% of time
        negative_codes = [c for c in code_to_indices.keys() if c != anchor_code]
        if not negative_codes:
            continue
        
        if np.random.random() < 0.4:
            # Hard negative: same writer, different code
            same_writer_codes_diff = [(code, writer) for code, writer in code_writer_to_indices.keys()
                                     if writer == anchor_writer and code != anchor_code]
            if same_writer_codes_diff:
                neg_code, neg_writer = random.choice(same_writer_codes_diff)
                negative_idx = np.random.choice(code_writer_to_indices[(neg_code, neg_writer)])
            else:
                # Fallback: any different code
                negative_code = np.random.choice(negative_codes)
                negative_idx = np.random.choice(code_to_indices[negative_code])
        else:
            # Semi-hard negative: common codes (more samples = more confusable)
            # Weight by frequency
            code_weights = [len(code_to_indices[c]) for c in negative_codes]
            total_weight = sum(code_weights)
            code_probs = [w / total_weight for w in code_weights]
            negative_code = np.random.choice(negative_codes, p=code_probs)
            negative_idx = np.random.choice(code_to_indices[negative_code])
        
        # Get base samples
        anchor_base = X[anchor_idx]
        positive_base = X[positive_idx]
        negative_base = X[negative_idx]
        
        # Apply augmentation (70% probability - helps both cross-writer and within-writer)
        if use_augmentation and np.random.random() > 0.3:
            anchor_augmented = augment_stroke_writer_aware(anchor_base, num_augmentations=1)
            anchor_sample = anchor_augmented[0] if anchor_augmented else anchor_base
            
            positive_augmented = augment_stroke_writer_aware(positive_base, num_augmentations=1)
            positive_sample = positive_augmented[0] if positive_augmented else positive_base
        else:
            anchor_sample = anchor_base
            positive_sample = positive_base
        
        triplets.append([
            anchor_sample,
            positive_sample,
            negative_base
        ])
        
        if len(triplets) >= max_triplets:
            break
    
    if len(triplets) == 0:
        return np.array([]), np.array([])
    
    triplets = np.array(triplets)
    labels = np.zeros((len(triplets), 1), dtype=np.float32)
    
    cross_writer_pct = (cross_writer_count / len(triplets) * 100) if len(triplets) > 0 else 0
    print(f"Created {len(triplets)} triplets for training (with writer-aware augmentation)")
    print(f"  - {cross_writer_count} ({cross_writer_pct:.1f}%) cross-writer positives")
    print(f"  - {len(triplets) - cross_writer_count} ({100-cross_writer_pct:.1f}%) same-writer positives")
    
    return triplets, labels





# Custom L2 Normalization Layer (avoids Lambda serialization issues)
class L2Normalize(keras.layers.Layer):
    """Custom layer for L2 normalization"""
    def __init__(self, axis=1, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        import tensorflow as tf
        return tf.nn.l2_normalize(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(L2Normalize, self).get_config()
        config.update({"axis": self.axis})
        return config

def build_stknet_encoder(embedding_dim=128):

    encoder_input = Input(shape=(64, 6), name='encoder_input')
    x = Reshape((64, 6, 1))(encoder_input)
    
    # STKNet Conv2D layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling2D((2, 1), name='pool1')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D((2, 1), name='pool2')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = Dropout(0.3)(x)
    embedding = Dense(embedding_dim, activation=None, name='embedding')(x)
    
    # L2 normalization for stable distance computation using custom layer
    embedding_normalized = L2Normalize(axis=1, name='l2_normalize')(embedding)
    
    model = Model(encoder_input, embedding_normalized)
    return model

def build_siamese_model(encoder):

    input1 = Input(shape=(64, 6), name='input1')
    input2 = Input(shape=(64, 6), name='input2')
    
    # Use the same encoder for both inputs (shared weights)
    emb1 = encoder(input1)
    emb2 = encoder(input2)
    
    # Concatenate embeddings for contrastive loss computation
    from tensorflow.keras.layers import Concatenate
    merged = Concatenate(axis=1)([emb1, emb2])
    
    model = Model([input1, input2], merged)
    return model

# ===== MAIN =====
if __name__ == "__main__":
    # ===== CONFIGURATION =====
    SKIP_TRAINING = False  # Set to True to skip training and load from saved model
    
    # Hard Negative Mining Configuration
    USE_HARD_MINING = True  # Use iterative hard negative mining
    MINING_STRATEGY = 'semi-hard'  # Options: 'hard', 'semi-hard', 'mixed'
    CONTRASTIVE_MARGIN = 1.0  # Margin for contrastive loss
    BATCH_SIZE = 64  # Pairs per batch (half positive, half negative)
    
    # Load data
    data_dir = "/Users/ballu_macbookpro/Downloads/Hanzi_Recognition_Project/Training"
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    print(f"Found {len(excel_files)} Excel files")
    
    df_list = []
    for f in excel_files:
        try:
            df_list.append(pd.read_excel(f))
        except:
            pass
    
    df = pd.concat(df_list, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    
    print(f"Total rows: {len(df)}")
    
    # Build stroke samples with character-level metadata
    (X_strokes, y_codes, stroke_ids, writers, char_nrs, 
     hanzis, char_to_strokes) = build_stroke_samples_from_df(df)
    
    X_strokes = np.array(X_strokes, dtype=np.float32)
    writers = np.array(writers)
    char_nrs = np.array(char_nrs)
    hanzis = np.array(hanzis)
    
    print(f"X_strokes shape: {X_strokes.shape}")
    print(f"Total unique Freeman codes (before filtering): {len(set(y_codes))}")
    print(f"Total unique writers: {len(set(writers))}")
    print(f"Total unique Hanzi: {len(set(hanzis))}")
    
    # ===== PHASE 1: DATA CURATION & CLASS BALANCING =====
    print("\n" + "="*60)
    print("PHASE 1: DATA CURATION & CLASS BALANCING")
    print("="*60)
    
    # Filter rare classes (classes with < MIN_SAMPLES samples)
    MIN_SAMPLES_PER_CLASS = 5
    print(f"\nFiltering classes with < {MIN_SAMPLES_PER_CLASS} samples...")
    
    from collections import Counter
    code_counts = Counter(y_codes)
    valid_codes = {code for code, count in code_counts.items() if count >= MIN_SAMPLES_PER_CLASS}
    
    # Filter dataset to only include valid classes
    valid_mask = np.array([code in valid_codes for code in y_codes])
    X_strokes_filtered = X_strokes[valid_mask]
    y_codes_filtered = [y_codes[i] for i in range(len(y_codes)) if valid_mask[i]]
    stroke_ids_filtered = [stroke_ids[i] for i in range(len(stroke_ids)) if valid_mask[i]]
    writers_filtered = writers[valid_mask]
    char_nrs_filtered = char_nrs[valid_mask]
    hanzis_filtered = hanzis[valid_mask]
    
    removed_classes = len(set(y_codes)) - len(valid_codes)
    removed_samples = len(y_codes) - len(y_codes_filtered)
    
    print(f"  Removed {removed_classes} rare classes")
    print(f"  Removed {removed_samples} samples ({removed_samples/len(y_codes)*100:.1f}%)")
    print(f"  Remaining classes: {len(valid_codes)}")
    print(f"  Remaining samples: {len(y_codes_filtered)}")
    
    # ===== STRATIFIED SPLIT BY FREEMAN CODE =====
    print("\nUsing stratified split to ensure all classes in train/test...")
    
    # Stratified split ensures every Freeman code appears in both train and test
    X_train, X_test, y_train, y_test, ids_train, ids_test, \
    writers_train, writers_test, char_nrs_train, char_nrs_test, \
    hanzis_train, hanzis_test = train_test_split(
        X_strokes_filtered, y_codes_filtered, stroke_ids_filtered, 
        writers_filtered, char_nrs_filtered, hanzis_filtered,
        test_size=0.2, 
        stratify=y_codes_filtered,  # KEY: Stratify by Freeman code
        random_state=42
    )
    
    # Verify stratification
    train_classes = len(set(y_train))
    test_classes = len(set(y_test))
    print(f"  Train classes: {train_classes}")
    print(f"  Test classes: {test_classes}")
    print(f"  Overlap: {len(set(y_train) & set(y_test))} classes")
    print(f" All classes appear in both splits!" if train_classes == test_classes else "  ⚠ Warning: Class mismatch!")
    
    print(f"\nStroke-level split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Build encoder with larger embedding for better cross-writer generalization
    print("\nBuilding STKNet encoder...")
    encoder = build_stknet_encoder(embedding_dim=256)  # Increased from 128 to 256
    
    if not SKIP_TRAINING:
        encoder.summary()
    
        # ===== CONTRASTIVE LOSS TRAINING WITH HARD NEGATIVE MINING =====
        print("\n" + "="*60)
        print("CONTRASTIVE LOSS WITH ITERATIVE HARD NEGATIVE MINING")
        print("="*60)
        print(f"Mining Strategy: {MINING_STRATEGY}")
        print(f"Margin: {CONTRASTIVE_MARGIN}")
        print(f"Batch Size: {BATCH_SIZE} pairs per batch")
        
        if USE_HARD_MINING:
            
            # Build siamese model for contrastive learning
            siamese_model = build_siamese_model(encoder)
            
            # Compile with contrastive loss
            optimizer = keras.optimizers.Adam(learning_rate=0.0005)
            siamese_model.compile(
                optimizer=optimizer,
                loss=contrastive_loss(margin=CONTRASTIVE_MARGIN)
            )
            
            # Add callbacks
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                HardNegativeMiningCallback(log_frequency=1),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Create online pair generator with hard negative mining
            print("Initializing online pair generator with iterative hard negative mining...")
            train_generator = create_contrastive_pairs_generator(
                X_train, y_train, writers_train, encoder,
                batch_size=BATCH_SIZE,
                use_augmentation=True,
                mining_strategy=MINING_STRATEGY,
                margin=CONTRASTIVE_MARGIN,
                memory_bank_size=5000
            )
            
            # Calculate steps per epoch
            steps_per_epoch = min(500, len(X_train) // BATCH_SIZE)
            
            print(f"\nTraining with online hard negative mining for up to 4 epochs")
            print(f"Steps per epoch: {steps_per_epoch}")
            print("Mining adapts iteratively as embeddings improve...\n")
            
            # Train with generator - use validation_split approach instead
            # Generate training data in batches
            print("Starting iterative training with hard negative mining...")
            num_epochs = 2
            for epoch in range(num_epochs):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"{'='*60}")
                
                epoch_losses = []
                for step in range(steps_per_epoch):
                    # Get next batch from generator
                    batch_data, batch_labels = next(train_generator)
                    
                    # Train on batch
                    loss = siamese_model.train_on_batch(batch_data, batch_labels)
                    epoch_losses.append(loss)
                    
                    if (step + 1) % 100 == 0:
                        avg_loss = np.mean(epoch_losses[-100:])
                        print(f"  Step {step+1}/{steps_per_epoch} - Loss: {avg_loss:.4f}")
                
                epoch_loss = np.mean(epoch_losses)
                print(f"\nEpoch {epoch+1} - Average Loss: {epoch_loss:.4f}")
                
                # Early stopping check (simple version)
                if epoch > 3 and len(callbacks[0].epoch_losses) > 0:
                    if callbacks[0].epoch_losses[-1] < 0.01:  # Very low loss
                        print("\nEarly stopping: Loss converged")
                        break
                
                # Log to callback
                callbacks[0].on_epoch_end(epoch, {'loss': epoch_loss})
            
            # Create history-like object
            class SimpleHistory:
                def __init__(self):
                    self.history = {'loss': callbacks[0].epoch_losses}
            
            history = SimpleHistory()
            
            print(f"\nTraining complete!")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            if 'val_loss' in history.history:
                print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
        else:
            # Fallback to static triplet generation (original method)
            print("\nUsing static triplet generation (no hard mining)...")
            triplets, triplet_labels = create_stroke_triplets(
                X_train, y_train, writers_train, max_triplets=50000
            )
            
            if len(triplets) == 0:
                print("⚠ Error: Could not generate triplets")
                exit(1)
            
            # Convert triplets to pairs for contrastive loss
            pairs_input1 = []
            pairs_input2 = []
            pair_labels = []
            
            for triplet in triplets:
                # Positive pair
                pairs_input1.append(triplet[0])  # anchor
                pairs_input2.append(triplet[1])  # positive
                pair_labels.append(1)
                
                # Negative pair
                pairs_input1.append(triplet[0])  # anchor
                pairs_input2.append(triplet[2])  # negative
                pair_labels.append(0)
            
            pairs_input1 = np.array(pairs_input1)
            pairs_input2 = np.array(pairs_input2)
            pair_labels = np.array(pair_labels).reshape(-1, 1)
            
            siamese_model = build_siamese_model(encoder)
            optimizer = keras.optimizers.Adam(learning_rate=0.0005)
            siamese_model.compile(
                optimizer=optimizer,
                loss=contrastive_loss(margin=CONTRASTIVE_MARGIN)
            )
            
            history = siamese_model.fit(
                [pairs_input1, pairs_input2],
                pair_labels,
                validation_split=0.1,
                epochs=10,
                batch_size=32,
                verbose=1
            )
            
            print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
    
        # Extract the trained encoder (weights are shared, so encoder is already updated)
        encoder.save("stknet_encoder_knn.keras")
        print("\nSaved encoder")
    else:
        print("\n" + "="*60)
        print("SKIPPING TRAINING - LOADING SAVED MODEL")
        print("="*60)
    
    # Load the saved encoder (either from training or from previous run)
    if os.path.exists("stknet_encoder_knn.keras"):
        # Load model with custom L2Normalize layer
        encoder = keras.models.load_model(
            "stknet_encoder_knn.keras",
            custom_objects={'L2Normalize': L2Normalize}
        )
        print("Loaded saved encoder from stknet_encoder_knn.keras")
    else:
        print("Warning: No saved model found. Please run with SKIP_TRAINING=False first.")
        exit(1)
    
    # Build library from training data
    print("\nBuilding stroke library...")
    train_embeddings = encoder.predict(X_train, verbose=0)
    
    # Check for NaN values
    if np.isnan(train_embeddings).any():
        print("NaN values detected in embeddings, replacing with zeros")
        train_embeddings = np.nan_to_num(train_embeddings, nan=0.0)
    
    # Use k=5 for voting-based prediction (more robust than k=1)
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(train_embeddings)
    print(f"Library built with {len(train_embeddings)} stroke embeddings (k=5 voting)")

    
    # ===== BUILD CANONICAL CHARACTER LIBRARY =====

    print("BUILDING CANONICAL CHARACTER LIBRARY")

    
    # Group ALL data by character to find canonical patterns (not just training)
    char_patterns = {}  # hanzi -> list of freeman_sequences
    
    for (writer, char_nr), stroke_indices in char_to_strokes.items():
        # Get Freeman codes for this character instance
        freeman_codes = []
        for idx in stroke_indices:
            if idx < len(y_codes):
                freeman_codes.append(y_codes[idx])
        
        if freeman_codes:
            # Assemble full character pattern
            pattern = ''.join(freeman_codes)
            hanzi = hanzis[stroke_indices[0]]
            
            if hanzi not in char_patterns:
                char_patterns[hanzi] = []
            char_patterns[hanzi].append(pattern)
    
    print(f"Found patterns for {len(char_patterns)} unique characters")
    
    # OPTIMIZATION: Compute ALL stroke embeddings once upfront
    print("Computing all stroke embeddings (this may take a few minutes)...")
    all_stroke_embeddings = encoder.predict(X_strokes, batch_size=256, verbose=1)
    
    # Check for NaN
    if np.isnan(all_stroke_embeddings).any():
        print("⚠ Warning: NaN values in embeddings, replacing with zeros")
        all_stroke_embeddings = np.nan_to_num(all_stroke_embeddings, nan=0.0)
    
    print("All embeddings computed")
    
    # Find most common (canonical) pattern for each character
    canonical_library = {}
    canonical_embeddings_list = []
    canonical_hanzis_list = []
    
    # Build index: pattern -> list of (writer, char_nr) for fast lookup
    pattern_to_instances = {}
    for (writer, char_nr), stroke_indices in char_to_strokes.items():
        instance_codes = [y_codes[idx] for idx in stroke_indices if idx < len(y_codes)]
        if instance_codes:
            pattern = ''.join(instance_codes)
            if pattern not in pattern_to_instances:
                pattern_to_instances[pattern] = []
            pattern_to_instances[pattern].append((writer, char_nr, stroke_indices))
    
    print("Building canonical embeddings...")
    for hanzi, patterns in char_patterns.items():
        # Count pattern frequencies
        from collections import Counter
        pattern_counts = Counter(patterns)
        # Select most common pattern as canonical
        canonical_pattern = pattern_counts.most_common(1)[0][0]
        canonical_library[hanzi] = canonical_pattern
        
        # Find instances with this canonical pattern
        if canonical_pattern in pattern_to_instances:
            # Use first instance as representative
            _, _, stroke_indices = pattern_to_instances[canonical_pattern][0]
            
            # Get embeddings for these strokes (already computed!)
            embeddings = [all_stroke_embeddings[idx] for idx in stroke_indices if idx < len(all_stroke_embeddings)]
            
            if embeddings:
                # Average embeddings for this character
                char_embedding = np.mean(embeddings, axis=0)
                canonical_embeddings_list.append(char_embedding)
                canonical_hanzis_list.append(hanzi)
    
    canonical_embeddings = np.array(canonical_embeddings_list)
    
    # Check for NaN in canonical embeddings
    if len(canonical_embeddings) > 0 and np.isnan(canonical_embeddings).any():
        print("Warning: NaN in canonical embeddings, replacing with zeros")
        canonical_embeddings = np.nan_to_num(canonical_embeddings, nan=0.0)
    
    print(f"Built canonical library with {len(canonical_library)} characters")
    print(f"Created {len(canonical_embeddings)} canonical character embeddings")
    
    # Build character-level KNN index
    if len(canonical_embeddings) > 0:
        char_knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        char_knn.fit(canonical_embeddings)
        print(f"Character-level KNN index ready")
    else:
        char_knn = None
        print("No canonical embeddings created")
    
    # Evaluate stroke-level accuracy
    print("EVALUATING STROKE-LEVEL PREDICTION")

    test_embeddings = encoder.predict(X_test, verbose=0)
    
    # Check for NaN in test embeddings
    if np.isnan(test_embeddings).any():
        print("Warning: NaN in test embeddings, replacing with zeros")
        test_embeddings = np.nan_to_num(test_embeddings, nan=0.0)
    
    # Find k=5 nearest neighbors for voting
    distances, indices = knn.kneighbors(test_embeddings)
    
    # Distance-weighted voting for each test stroke
    y_pred_test = []
    correct = 0
    low_confidence_count = 0
    confidence_threshold = 0.85  # d1/d2 ratio threshold
    
    for i in range(len(indices)):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]
        
        # Confidence check: ratio of closest to second-closest distance
        if len(neighbor_distances) >= 2 and neighbor_distances[0] > 0:
            confidence_ratio = neighbor_distances[0] / neighbor_distances[1]
            if confidence_ratio > confidence_threshold:
                # Low confidence - still predict but track it
                low_confidence_count += 1
        
        # Distance-weighted voting (inverse distance weights)
        # Avoid division by zero with small epsilon
        epsilon = 1e-10
        weights = 1.0 / (neighbor_distances + epsilon)
        
        # Collect votes from k neighbors
        vote_counts = {}
        vote_weights = {}
        for idx, weight in zip(neighbor_indices, weights):
            code = y_train[idx]
            vote_counts[code] = vote_counts.get(code, 0) + 1
            vote_weights[code] = vote_weights.get(code, 0) + weight
        
        # Predict based on weighted votes
        predicted_code = max(vote_weights, key=vote_weights.get)
        y_pred_test.append(predicted_code)
        
        if predicted_code == y_test[i]:
            correct += 1
    
    stroke_accuracy = correct / len(y_test) * 100
    low_conf_pct = (low_confidence_count / len(y_test) * 100) if len(y_test) > 0 else 0
    

    print(f"STROKE-LEVEL RESULTS (k=5 Distance-Weighted Voting):")
    print(f"Exact Match Accuracy: {stroke_accuracy:.2f}%")
    print(f"Correct: {correct}/{len(y_test)}")
    print(f"Low Confidence Predictions: {low_confidence_count} ({low_conf_pct:.1f}%)")
    print(f"  (distance ratio d1/d2 > {confidence_threshold})")

    
    # Sample predictions - emphasize Freeman codes (actual prediction target)
    print("\nSample Stroke Predictions:")
    print(f"{'True Freeman':<20} | {'Pred Freeman':<20} | {'Match':<8} | {'From Char':<10}")
    print("-" * 70)
    for i in range(min(10, len(indices))):
        nearest_idx = indices[i, 0]
        true_code = y_test[i]
        pred_code = y_pred_test[i]
        hanzi = hanzis_test[i]
        match = "Right" if true_code == pred_code else "Wrong"
        print(f"{true_code:<20} | {pred_code:<20} | {match:<8} | {hanzi:<10}")
    
    # ===== CHARACTER-LEVEL EVALUATION =====
    print("EVALUATING CHARACTER-LEVEL RECOGNITION")
    
    
    # Create mapping for test strokes
    # We need to map test stroke indices to their predictions
    test_stroke_idx_to_pred = {}
    test_stroke_idx_to_true = {}
    test_stroke_idx_to_id = {}
    
    for i, idx in enumerate(ids_test):
        test_stroke_idx_to_pred[idx] = y_pred_test[i]
        test_stroke_idx_to_true[idx] = y_test[i]
        test_stroke_idx_to_id[idx] = i
    
    # Build character-level test set from char_to_strokes
    # Include characters with at least 50% of their strokes in test set
    char_test_set = {}
    for (writer, char_nr), stroke_indices in char_to_strokes.items():
        if len(stroke_indices) == 0:
            continue
        
        strokes_in_test = sum(1 for idx in stroke_indices if idx in test_stroke_idx_to_true)
        coverage = strokes_in_test / len(stroke_indices)
        
        # Include if at least 50% of strokes are in test set (balanced quality control)
        if coverage >= 0.5:
            char_test_set[(writer, char_nr)] = stroke_indices
    
    # Build Freeman code -> Hanzi mapping from ALL characters AND canonical library
    print("\nBuilding Freeman code -> Hanzi mapping...")
    freeman_to_hanzi_map = {}
    
    # First, add canonical patterns (highest priority)
    for hanzi, pattern in canonical_library.items():
        freeman_to_hanzi_map[pattern] = hanzi
    
    # Then add all training character instances for coverage
    for (writer, char_nr), stroke_indices in char_to_strokes.items():
        # Assemble Freeman code for this character
        freeman_codes = []
        for idx in stroke_indices:
            if idx < len(np.array(y_codes)):
                freeman_codes.append(np.array(y_codes)[idx])
        
        if freeman_codes and stroke_indices:
            assembled_freeman = ''.join(freeman_codes)
            hanzi = hanzis[stroke_indices[0]]
            # Only add if not already in map (canonical takes priority)
            if assembled_freeman not in freeman_to_hanzi_map:
                freeman_to_hanzi_map[assembled_freeman] = hanzi
    
    print(f"Created mapping for {len(freeman_to_hanzi_map)} unique character patterns")
    print(f"  - {len(canonical_library)} canonical patterns")
    print(f"  - {len(freeman_to_hanzi_map) - len(canonical_library)} variant patterns")
    
    # WITHIN-WRITER: Evaluate on test set (same writer, different characters)
    if len(char_test_set) > 0:
        evaluate_character_level(
            char_test_set,
            [test_stroke_idx_to_true.get(i, "") for i in range(max(test_stroke_idx_to_true.keys()) + 1)],
            [test_stroke_idx_to_pred.get(i, "") for i in range(max(test_stroke_idx_to_pred.keys()) + 1)],
            writers,
            hanzis,
            freeman_to_hanzi_map,
            all_stroke_embeddings=all_stroke_embeddings,
            canonical_embeddings=canonical_embeddings,
            canonical_hanzis_list=canonical_hanzis_list,
            scenario_name="WITHIN-WRITER (Random Split)"
        )
    
    # CROSS-WRITER: Split by writer (use multiple writers)
    unique_writers = sorted(list(set(writers)))
    num_writers = len(unique_writers)
    
    if num_writers >= 3:
        # Use 70% of writers for training, 30% for testing
        num_train_writers = max(1, int(num_writers * 0.7))
        train_writers = unique_writers[:num_train_writers]
        test_writers = unique_writers[num_train_writers:]
        
        print(f"\n{'='*60}")
        print(f"CROSS-WRITER SETUP:")
        print(f"Total writers: {num_writers}")
        print(f"Train writers: {train_writers} ({len(train_writers)} writers)")
        print(f"Test writers: {test_writers} ({len(test_writers)} writers)")
        print(f"{'='*60}")
        
        # Get strokes for cross-writer split (multiple writers)
        train_writer_mask = np.isin(writers, train_writers)
        test_writer_mask = np.isin(writers, test_writers)
        
        X_train_cw = X_strokes[train_writer_mask]
        y_train_cw = np.array(y_codes)[train_writer_mask]
        ids_train_cw = np.array(stroke_ids)[train_writer_mask]
        
        X_test_cw = X_strokes[test_writer_mask]
        y_test_cw = np.array(y_codes)[test_writer_mask]
        ids_test_cw = np.array(stroke_ids)[test_writer_mask]
        hanzis_test_cw = hanzis[test_writer_mask]
        
        print(f"Cross-writer train strokes: {len(X_train_cw)}")
        print(f"Cross-writer test strokes: {len(X_test_cw)}")
        
        if len(X_train_cw) > 0 and len(X_test_cw) > 0:
            # Rebuild KNN with cross-writer train set (k=5 for voting)
            train_embeddings_cw = encoder.predict(X_train_cw, verbose=0)
            
            # Check for NaN in cross-writer embeddings
            if np.isnan(train_embeddings_cw).any():
                print("⚠ Warning: NaN in cross-writer embeddings, replacing with zeros")
                train_embeddings_cw = np.nan_to_num(train_embeddings_cw, nan=0.0)
            
            knn_cw = NearestNeighbors(n_neighbors=5, metric='cosine')  # k=5 voting
            knn_cw.fit(train_embeddings_cw)
            
            # Predict for cross-writer test set
            test_embeddings_cw = encoder.predict(X_test_cw, verbose=0)
            
            # Check for NaN in cross-writer test embeddings
            if np.isnan(test_embeddings_cw).any():
                print("⚠ Warning: NaN in cross-writer test embeddings, replacing with zeros")
                test_embeddings_cw = np.nan_to_num(test_embeddings_cw, nan=0.0)
            distances_cw, indices_cw = knn_cw.kneighbors(test_embeddings_cw)
            
            # Distance-weighted voting for cross-writer predictions
            y_pred_test_cw = []
            correct_cw = 0
            for i in range(len(indices_cw)):
                neighbor_indices = indices_cw[i]
                neighbor_distances = distances_cw[i]
                
                # Distance-weighted voting
                epsilon = 1e-10
                weights = 1.0 / (neighbor_distances + epsilon)
                
                vote_weights = {}
                for idx, weight in zip(neighbor_indices, weights):
                    code = y_train_cw[idx]
                    vote_weights[code] = vote_weights.get(code, 0) + weight
                
                predicted_code = max(vote_weights, key=vote_weights.get)
                y_pred_test_cw.append(predicted_code)
                if predicted_code == y_test_cw[i]:
                    correct_cw += 1
            
            stroke_accuracy_cw = (correct_cw / len(y_test_cw) * 100) if len(y_test_cw) > 0 else 0
            
            print(f"\nCross-Writer Stroke-Level Accuracy: {stroke_accuracy_cw:.2f}%")
            
            # Character-level for cross-writer
            test_cw_stroke_idx_to_pred = {}
            test_cw_stroke_idx_to_true = {}
            for i, idx in enumerate(ids_test_cw):
                test_cw_stroke_idx_to_pred[idx] = y_pred_test_cw[i]
                test_cw_stroke_idx_to_true[idx] = y_test_cw[i]
            
            char_test_cw = {}
            for (writer, char_nr), stroke_indices in char_to_strokes.items():
                if writer in test_writers:
                    all_in_test = all(idx in test_cw_stroke_idx_to_true for idx in stroke_indices)
                    if all_in_test and len(stroke_indices) > 0:
                        char_test_cw[(writer, char_nr)] = stroke_indices
            
            if len(char_test_cw) > 0:
                evaluate_character_level(
                    char_test_cw,
                    [test_cw_stroke_idx_to_true.get(i, "") for i in range(max(test_cw_stroke_idx_to_true.keys()) + 1)],
                    [test_cw_stroke_idx_to_pred.get(i, "") for i in range(max(test_cw_stroke_idx_to_pred.keys()) + 1)],
                    writers,
                    hanzis,
                    freeman_to_hanzi_map,
                    all_stroke_embeddings=all_stroke_embeddings,
                    canonical_embeddings=canonical_embeddings,
                    canonical_hanzis_list=canonical_hanzis_list,
                    scenario_name="CROSS-WRITER"
                )
    
    
    print(f"Stroke-Level Accuracy (Random Split): {stroke_accuracy:.2f}%")
    print(f"\nCanonical Character Library:")
    print(f"  - {len(canonical_library)} unique character patterns")


