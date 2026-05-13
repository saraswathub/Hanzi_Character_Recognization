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

# Global Configuration
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Freeman direction mapping (0-7)
DIR_MAP = {0: (1,0), 1: (1,1), 2: (0,1), 3: (-1,1), 4: (-1,0), 5: (-1,-1), 6: (0,-1), 7: (1,-1)}

def edit_distance(s1, s2):
    """Levenshtein distance for fuzzy stroke matching."""
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
    """Normalize freeman strings to integer sequences."""
    return [int(c) for c in str(s) if c.isdigit() and int(c) <= 7]

def generate_trajectory(codes, start=(0, 0)):
    """Reconstruct 2D path from discrete Freeman codes."""
    x, y = start
    pts = [(x, y, 1.0, 0.0)]
    for c in codes:
        dx, dy = DIR_MAP.get(c, (0, 0))
        x += dx; y += dy
        pts.append((x, y, 1.0, 1.0 if c == 8 else 0.0))
    return np.array(pts, dtype=np.float32)

def build_research_dataset(df):
    """Process raw Excel logs into stroke-level and character-level structures."""
    strokes, labels, writers, hanzis, absolute_trajectories = [], [], [], [], []
    char_to_strokes = {}
    stroke_idx = 0
    
    for (writer, char_nr), g in df.groupby(['writer', 'character_nr']):
        g = g.sort_values('strock_number')
        current_pos = (0.0, 0.0)
        char_key = (writer, char_nr)
        char_to_strokes[char_key] = []
        
        for _, row in g.iterrows():
            code_str = str(row['strock_label_freeman'])
            codes = parse_freeman(code_str)
            if not codes: continue
            
            traj = generate_trajectory(codes, start=current_pos)
            absolute_trajectories.append(traj.copy())
            current_pos = (traj[-1, 0], traj[-1, 1])
            
            # Resampling to 64 points for STKNet input
            p_min, p_max = traj[:, :2].min(axis=0), traj[:, :2].max(axis=0)
            res_size = np.maximum(p_max - p_min, 1e-6)
            norm_xy = (traj[:, :2] - (p_min + res_size/2)) / np.max(res_size)
            
            idx_old, idx_new = np.arange(len(traj)), np.linspace(0, len(traj)-1, 64)
            resampled_xy = np.stack([np.interp(idx_new, idx_old, norm_xy[:, i]) for i in range(2)], axis=1)
            n_idx = np.clip(np.round(idx_new).astype(int), 0, len(traj)-1)
            
            # Assemble 6-channel input (XY + Pen + Hyphen + Velocity)
            v = np.zeros((64, 6), dtype=np.float32)
            v[:, :2], v[:, 2:4] = resampled_xy, traj[n_idx, 2:4]
            v[1:, 4], v[1:, 5] = resampled_xy[1:, 0] - resampled_xy[:-1, 0], resampled_xy[1:, 1] - resampled_xy[:-1, 1]
            
            strokes.append(v); labels.append(code_str); writers.append(writer); hanzis.append(str(row['hanzi']))
            char_to_strokes[char_key].append(stroke_idx); stroke_idx += 1
            
    return (np.array(strokes), labels, writers, hanzis, char_to_strokes, absolute_trajectories)

def build_stknet(input_shape=(64, 6), embedding_dim=256):
    """STKNet Encoder Architecture."""
    inp = Input(shape=input_shape)
    x = Reshape((64, 6, 1))(inp)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = MaxPooling2D((2, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = MaxPooling2D((2, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x); x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x); x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x); x = Dropout(0.4)(x)
    emb = Dense(embedding_dim, activation=None)(x)
    norm_emb = Lambda(lambda v: tf.nn.l2_normalize(v, axis=1))(emb)
    return Model(inp, norm_emb)

def extract_structural_grid(stroke_codes_list, trajectories, grid_size=12):
    """Directional Structural Grid (DSG) feature extraction."""
    if not trajectories: return np.zeros(grid_size * grid_size * 8)
    pts = np.concatenate(trajectories, axis=0)
    p_min, p_max = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    w, h = max(p_max[0] - p_min[0], 1e-6), max(p_max[1] - p_min[1], 1e-6)
    grid = np.zeros((grid_size, grid_size, 8))
    
    for codes, orig_traj in zip(stroke_codes_list, trajectories):
        path_codes = parse_freeman(codes)
        path = generate_trajectory(path_codes)
        if len(path) < 2: continue
        
        c_min, c_max = path[:, :2].min(axis=0), path[:, :2].max(axis=0)
        cw, ch = max(c_max[0]-c_min[0], 1e-6), max(c_max[1]-c_min[1], 1e-6)
        o_min, o_max = orig_traj[:, :2].min(axis=0), orig_traj[:, :2].max(axis=0)
        ow, oh = max(o_max[0]-o_min[0], 1e-6), max(o_max[1]-o_min[1], 1e-6)
        warped = (path[:, :2] - c_min) / [cw, ch] * [ow, oh] + o_min
        
        for j in range(len(path_codes)):
            mx, my = (warped[j] + warped[j+1]) / 2
            gx = int(max(0, min(grid_size-1, (mx - p_min[0]) / w * grid_size)))
            gy = int(max(0, min(grid_size-1, (my - p_min[1]) / h * grid_size)))
            grid[gy, gx, path_codes[j]] += 1.0
            
    vec = grid.flatten(); norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec

if __name__ == "__main__":
    print("[Pipeline] Initializing Hanzi Recognition...")
    data_files = glob.glob("Training/*.xlsx")
    dfs = [sheet for f in data_files for _, sheet in pd.read_excel(f, sheet_name=None).items() if len(sheet)>0]
    master_df = pd.concat(dfs, ignore_index=True); master_df.columns = [c.strip() for c in master_df.columns]
    master_df = master_df.drop_duplicates(subset=['writer', 'character_nr', 'strock_number'])
    
    X_stk, y_labels, writers, hanzis, char_to_strokes, X_abs = build_research_dataset(master_df)
    
    # Model Initialization
    stknet = build_stknet()
    if os.path.exists("stknet_encoder_knn.keras"):
        stknet.load_weights("stknet_encoder_knn.keras")
        print("[Model] STKNet Weights Loaded successfully.")
    
    # Feature Synthesis
    print("[Extraction] Generating embeddings and DSG features...")
    stroke_embs = stknet.predict(X_stk, batch_size=256)
    char_data = []
    for (w, cn), s_idx in char_to_strokes.items():
        char_data.append({'writer': w, 'hanzi': hanzis[s_idx[0]], 'codes': [y_labels[i] for i in s_idx], 
                          'trajs': [X_abs[i] for i in s_idx], 'indices': s_idx})

    dsg_features = np.array([extract_structural_grid(c['codes'], c['trajs']) for c in char_data])
    
    # Cross-Writer Evaluation Protocol (Train on W5, Test on others)
    train_w, test_w = [5], [1, 2, 3, 4]
    train_idx = [i for i, c in enumerate(char_data) if c['writer'] in train_w]
    test_idx = [i for i, c in enumerate(char_data) if c['writer'] in test_w]
    
    # Build Character Dictionary (Centroid-based)
    lib = defaultdict(list)
    for i in train_idx: lib[char_data[i]['hanzi']].append(dsg_features[i])
    lib_hz = sorted(list(lib.keys()))
    lib_embs = np.array([np.mean(lib[h], axis=0) for h in lib_hz])
    lib_embs /= np.linalg.norm(lib_embs, axis=1, keepdims=True)
    knn_char = NearestNeighbors(n_neighbors=5, metric='cosine').fit(lib_embs)
    
    # Performance Evaluation
    print("[Evaluation] Computing cross-writer synthesis metrics...")
    stroke_train_idx = [i for i, w in enumerate(writers) if w in train_w]
    stroke_test_idx = [i for i, w in enumerate(writers) if w in test_w]
    stroke_knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(stroke_embs[stroke_train_idx])
    dist_s, ind_s = stroke_knn.kneighbors(stroke_embs[stroke_test_idx])
    
    y_pred = list(y_labels); s_correct = 0
    for i, test_i in enumerate(stroke_test_idx):
        votes = defaultdict(float); weights = 1.0 / (dist_s[i] + 1e-10)
        for d_idx, w in zip(ind_s[i], weights):
            votes[y_labels[stroke_train_idx[d_idx]]] += w
        pred = max(votes, key=votes.get); y_pred[test_i] = pred
        if edit_distance(parse_freeman(pred), parse_freeman(y_labels[test_i])) <= 1: s_correct += 1
    
    test_dsg = np.array([extract_structural_grid([y_pred[idx] for idx in char_data[i]['indices']], char_data[i]['trajs']) for i in test_idx])
    test_dsg_norm = test_dsg / np.linalg.norm(test_dsg, axis=1, keepdims=True)
    _, ind_c = knn_char.kneighbors(test_dsg_norm)
    e2e_acc = sum(lib_hz[idx[0]] == char_data[test_idx[i]]['hanzi'] for i, idx in enumerate(ind_c)) / len(test_idx)

    print(f"\n[Final Benchmarks]")
    print(f"  Stroke Matching Accuracy: {s_correct/len(stroke_test_idx)*100:.2f}%")
    print(f"  Cross-Writer Generalization: {e2e_acc*100:.2f}%")
