import os
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Input, Flatten, Dropout, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# Setup and constants
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Directions for Freeman code (0-7)
DIRECTION_VECTORS = {0: (1,0), 1: (1,1), 2: (0,1), 3: (-1,1), 4: (-1,0), 5: (-1,-1), 6: (0,-1), 7: (1,-1)}

def edit_distance(s1, s2):
    """Fuzzy matching for stroke codes."""
    if len(s1) < len(s2): return edit_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]

def parse_freeman(s):
    return [int(c) for c in str(s) if c.isdigit() and int(c) <= 7]

def freeman_to_xy(codes, start=(0, 0)):
    x, y = start
    pts = [(x, y, 1.0, 0.0)]
    for c in codes:
        dx, dy = DIRECTION_VECTORS.get(c, (0, 0))
        x += dx; y += dy
        pts.append((x, y, 1.0, 1.0 if c == 8 else 0.0))
    return np.array(pts, dtype=np.float32)

# Data processing functions
def build_dataset(df):
    strokes, labels, writers, hanzis, absolute_trajectories = [], [], [], [], []
    char_to_strokes = {}
    stroke_idx = 0
    
    # Group by writer and character number to maintain sequence
    for (writer, char_nr), g in df.groupby(['writer', 'character_nr']):
        g = g.sort_values('strock_number')
        current_pos = (0.0, 0.0)
        char_key = (writer, char_nr)
        char_to_strokes[char_key] = []
        
        for _, row in g.iterrows():
            code_str = str(row['strock_label_freeman'])
            codes = parse_freeman(code_str)
            if not codes: continue
            
            traj = freeman_to_xy(codes, start=current_pos)
            absolute_trajectories.append(traj.copy())
            current_pos = (traj[-1, 0], traj[-1, 1])
            
            # Spatial normalization
            p_min, p_max = traj[:, :2].min(axis=0), traj[:, :2].max(axis=0)
            size = np.maximum(p_max - p_min, 1e-6)
            norm_xy = (traj[:, :2] - (p_min + size/2)) / np.max(size)
            
            # Resample to 64 points (Linear for XY, Nearest for Flags)
            idx_old = np.arange(len(traj))
            idx_new = np.linspace(0, len(traj)-1, 64)
            resampled_xy = np.stack([np.interp(idx_new, idx_old, norm_xy[:, i]) for i in range(2)], axis=1)
            n_idx = np.clip(np.round(idx_new).astype(int), 0, len(traj)-1)
            resampled_others = traj[n_idx, 2:4]
            
            # Feature channels: [X, Y, Pen, Hyphen, VX, VY]
            v = np.zeros((64, 6), dtype=np.float32)
            v[:, :2] = resampled_xy
            v[:, 2:4] = resampled_others
            v[1:, 4] = resampled_xy[1:, 0] - resampled_xy[:-1, 0]
            v[1:, 5] = resampled_xy[1:, 1] - resampled_xy[:-1, 1]
            
            strokes.append(v)
            labels.append(code_str)
            writers.append(writer)
            hanzis.append(str(row['hanzi']))
            char_to_strokes[char_key].append(stroke_idx)
            stroke_idx += 1
            
    return (np.array(strokes), labels, writers, hanzis, char_to_strokes, absolute_trajectories)

# Neural Network Architecture
def get_stknet_encoder(input_shape=(64, 6), embedding_dim=256):
    inp = Input(shape=input_shape)
    x = Reshape((64, 6, 1))(inp)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    emb = Dense(embedding_dim, activation=None)(x)
    norm_emb = Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(emb)
    return Model(inp, norm_emb)

# Directional Freeman Grid (DFG) Extraction
def extract_dfg(stroke_codes_list, trajectories, grid_size=12):
    if not trajectories: return np.zeros(grid_size * grid_size * 8)
    pts = np.concatenate(trajectories, axis=0)
    p_min, p_max = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    w, h = max(p_max[0] - p_min[0], 1e-6), max(p_max[1] - p_min[1], 1e-6)
    
    grid = np.zeros((grid_size, grid_size, 8))
    for codes, orig_traj in zip(stroke_codes_list, trajectories):
        path_codes = parse_freeman(codes)
        path = freeman_to_xy(path_codes)
        if len(path) < 2: continue
        
        curr_min, curr_max = path[:, :2].min(axis=0), path[:, :2].max(axis=0)
        cw, ch = max(curr_max[0]-curr_min[0], 1e-6), max(curr_max[1]-curr_min[1], 1e-6)
        o_min, o_max = orig_traj[:, :2].min(axis=0), orig_traj[:, :2].max(axis=0)
        ow, oh = max(o_max[0]-o_min[0], 1e-6), max(o_max[1]-o_min[1], 1e-6)
        
        # Mapping predicted path back to original character bounds
        warped = (path[:, :2] - curr_min) / [cw, ch] * [ow, oh] + o_min
        for j in range(len(path_codes)):
            mx, my = (warped[j] + warped[j+1]) / 2
            gx = int(max(0, min(grid_size-1, (mx - p_min[0]) / w * grid_size)))
            gy = int(max(0, min(grid_size-1, (my - p_min[1]) / h * grid_size)))
            grid[gy, gx, path_codes[j]] += 1.0
            
    vec = grid.flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec

if __name__ == "__main__":
    print("Loading datasets...")
    files = glob.glob("Training/*.xlsx")
    dfs = [sheet for f in files for _, sheet in pd.read_excel(f, sheet_name=None).items() if len(sheet)>0]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset=['writer', 'character_nr', 'strock_number'])
    
    X_stk, y_labels, writers, hanzis, char_to_strokes, X_abs = build_dataset(df)
    
    # Initialize and load STKNet
    encoder = get_stknet_encoder()
    if os.path.exists("stknet_encoder_knn.keras"):
        encoder.load_weights("stknet_encoder_knn.keras")
        print("Model weights loaded.")
    
    # Generate stroke embeddings
    stroke_embs = encoder.predict(X_stk, batch_size=256)
    
    # Prepare character-level features (DFG)
    char_data = []
    for (w, cn), s_idx in char_to_strokes.items():
        char_data.append({'writer': w, 'hanzi': hanzis[s_idx[0]], 
                          'codes': [y_labels[i] for i in s_idx], 
                          'trajs': [X_abs[i] for i in s_idx], 'indices': s_idx})

    dfg_feats = np.array([extract_dfg(c['codes'], c['trajs']) for c in char_data])

    # Evaluation Scenario (Reversed Cross-Writer)
    train_w, test_w = [5], [1, 2, 3, 4]
    train_idx = [i for i, c in enumerate(char_data) if c['writer'] in train_w]
    test_idx = [i for i, c in enumerate(char_data) if c['writer'] in test_w]
    
    # Build character library (Instance-Level)
    lib_embs = dfg_feats[train_idx]
    lib_labels = [char_data[i]['hanzi'] for i in train_idx]
    knn_char = NearestNeighbors(n_neighbors=5, metric='cosine').fit(lib_embs)
    
    # End-to-End Prediction Loop
    print("Evaluating recognition accuracy...")
    stroke_train_idx = [i for i, w in enumerate(writers) if w in train_w]
    stroke_test_idx = [i for i, w in enumerate(writers) if w in test_w]
    stroke_knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(stroke_embs[stroke_train_idx])
    dist_s, ind_s = stroke_knn.kneighbors(stroke_embs[stroke_test_idx])
    
    y_pred = list(y_labels)
    s_correct = 0
    for i, test_i in enumerate(stroke_test_idx):
        votes = defaultdict(float)
        weights = 1.0 / (dist_s[i] + 1e-10)
        for d_idx, w in zip(ind_s[i], weights):
            votes[y_labels[stroke_train_idx[d_idx]]] += w
        pred = max(votes, key=votes.get)
        y_pred[test_i] = pred
        if edit_distance(parse_freeman(pred), parse_freeman(y_labels[test_i])) <= 1:
            s_correct += 1
    
    # Character Recognition with Instance-Level Voting
    e2e_feats = np.array([extract_dfg([y_pred[idx] for idx in char_data[i]['indices']], char_data[i]['trajs']) for i in test_idx])
    e2e_embs = e2e_feats / np.linalg.norm(e2e_feats, axis=1, keepdims=True)
    dist_c, ind_c = knn_char.kneighbors(e2e_embs)
    
    e2e_correct = 0
    for i, idx_list in enumerate(ind_c):
        char_votes = defaultdict(float)
        weights = 1.0 / (dist_c[i] + 1e-10)
        for d_idx, w in zip(idx_list, weights):
            char_votes[lib_labels[d_idx]] += w
        pred_hz = max(char_votes, key=char_votes.get)
        if pred_hz == char_data[test_idx[i]]['hanzi']:
            e2e_correct += 1

    print(f"\nResults Summary:")
    print(f"  Stroke Accuracy:          {s_correct/len(stroke_test_idx)*100:.2f}%")
    print(f"  Cross Writer Accuracy:    {e2e_correct/len(test_idx)*100:.2f}%")
