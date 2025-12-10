# LidarRoofSegNet/trainModel.py
# ==========================================
# 0. IMPORTS
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MeanShift, DBSCAN

# Try importing HDBSCAN (available in scikit-learn >= 1.3)
try:
    from sklearn.cluster import HDBSCAN
    SKLEARN_HDBSCAN = True
except ImportError:
    SKLEARN_HDBSCAN = False
    print("WARNING: HDBSCAN not found in sklearn.cluster (requires scikit-learn >= 1.3).")

from typing import Tuple 

from tqdm import tqdm

# Try importing PyTorch Geometric, handle error gracefully if missing
try:
    from torch_geometric.nn import DynamicEdgeConv, global_max_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("WARNING: torch_geometric not found. DGCNN will fail to initialize.")

# Try importing Open3D for normal estimation
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("WARNING: Open3D not found. Normal estimation will be skipped.")

# Try importing DataAugmentation (local file)
try:
    from data_augmentation import DataAugmentation
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("WARNING: data_augmentation.py not found. Training will proceed without augmentation.")


print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0)}')


# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    """Global configuration for training and model parameters."""
    # Data Params
    input_dim = 6  # XYZ (3) + Normals (3). Set to 3 if O3D is missing.
    max_points = 4096
    
    # Model Params
    k_neighbors = 20
    emb_dim = 128
    output_dim = 64
    
    # Training Params
    batch_size = 16
    lr = 1e-3
    epochs = 50
    # First use GPU if available, else if MPS is available (Mac), else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    
    # Loss Params (Discriminative Loss)
    delta_v = 0.3  # Pull force margin
    delta_d = 1.5  # Push force margin
    alpha = 1.0    # Weight for pull
    beta = 1.0     # Weight for push
    gamma = 0.001  # Weight for regularization

    # Samopling Params
    sampling_method = 'fps'  # 'random' or 'fps' (farthest point sampling)

    # clustering methods
    clustering_method = 'hdbscan'  # 'mean_shift', 'dbscan', 'hdbscan'



# ==========================================
# 2. UTILITIES & PREPROCESSING
# ==========================================
def compute_normals(points, k=30):
    """
    Computes surface normals using Open3D.
    Args:
        points: (N, 3) numpy array
    Returns:
        features: (N, 6) numpy array [x, y, z, nx, ny, nz]
    """
    if not O3D_AVAILABLE:
        return points # Return just XYZ if Open3D is missing
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Hybrid search for robustness
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=k)
    )
    # Consistency check (orient towards Z-axis roughly for roofs)
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 100.]))
    
    normals = np.asarray(pcd.normals)
    return np.hstack((points, normals))

def normalize_pc(points):
    """Normalize point cloud to unit sphere at origin."""
    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    max_dist = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
    points[:, :3] /= max_dist
    return points


# ==========================================
# 3. DATASET
# ==========================================
class LiDARPointCloudDataset(Dataset):
    def __init__(self, base_dir, split="train", max_points=2048, sampling_method="random"):
        """
        Args:
            root_dir (str): Root directory containing the 'train_test_split' folder.
            split (str): One of 'train', 'val', or 'test'.
            max_points (int): Maximum number of points per cloud (for subsampling or padding).
            sampling_method (str): 'random' for random subsampling, 'fps' for Farthest Point Sampling.
        """
        self.base_dir = base_dir
        self.split = split
        self.max_points = max_points
        self.sampling_method = sampling_method
        
        # Determine internal mode: 'train' enables relabeling, others strictly eval
        self.mode = "train" if split == "train" else "eval"

        # Construct paths based on the requested structure:
        # data/train_test_split/points_{split}_n
        # base_dir = os.path.join(root_dir, "train_test_split")
        self.point_folder = os.path.join(base_dir, f"points_{split}_n")
        self.label_folder = os.path.join(base_dir, f"labels_{split}_n")

        # Ensure directories exist (or handle gracefully)
        if not os.path.exists(self.point_folder) or not os.path.exists(self.label_folder):
            print(f"WARNING: Data folders not found for split '{split}':\n  {self.point_folder}\n  {self.label_folder}")
            self.point_files = []
            self.label_files = []
        else:
            # List all available point files
            self.point_files = sorted([f for f in os.listdir(self.point_folder) if f.endswith(".txt")])
            self.label_files = sorted([f for f in os.listdir(self.label_folder) if f.endswith(".txt")])

            # Ensure matching point and label files
            assert len(self.point_files) == len(self.label_files), \
                f"Mismatch in points and labels count for split '{split}'."

    def _farthest_point_sampling(self, points, n_samples):
        """
        Performs Farthest Point Sampling (FPS).
        Selects points that are farthest from each other to ensure uniform coverage.
        
        Args:
            points: (N, 3) or (N, D) numpy array
            n_samples: int, number of points to select
        Returns:
            indices: (n_samples,) numpy array of selected indices
        """
        N, D = points.shape
        xyz = points[:, :3]  # Use only spatial coordinates for distance
        centroids = np.zeros((n_samples,), dtype=np.int64)
        
        # Initialize distances with infinity
        distance = np.ones((N,), dtype=np.float64) * 1e10
        
        # Select the first point randomly
        farthest = np.random.randint(0, N)
        
        for i in range(n_samples):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            
            # Calculate Euclidean distance from the current centroid to all points
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            
            # Update the minimum distance to any selected point for all points
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Select the point with the maximum distance to the set of selected points
            farthest = np.argmax(distance)
            
        return centroids

    def load_txt_file(self, file_path, num_features=3):
        """
        Loads a .txt file, converting string lines into a NumPy array of type float64.
        Assumes space/comma-separated values.
        """
        data = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    values = np.array(line.strip().replace(',', ' ').split(), dtype=np.float64)
                    if len(values) == num_features:
                        data.append(values)
                except ValueError:
                    continue  # Skip lines that cannot be converted
        
        if len(data) == 0:
            return np.zeros((0, num_features), dtype=np.float64)

        return np.array(data)

    def load_point_cloud(self, file_path):
        """Loads point cloud (XYZ) from a .txt file and returns a NumPy array."""
        return self.load_txt_file(file_path, num_features=3)  # Expecting [x, y, z]

    def load_labels(self, file_path, num_points):
        """Loads labels from a .txt file, ensuring it matches the number of points."""
        labels = self.load_txt_file(file_path, num_features=1).flatten()
        if len(labels) != num_points:
            print(f"Warning: {file_path} has {len(labels)} labels, expected {num_points}. Using zero-padding.")
            labels = np.pad(labels, (0, max(0, num_points - len(labels))), 'constant', constant_values=0)
        return labels

    def relabel_instances(self, labels):
        """
        Relabels instance IDs to be 0-indexed and returns the new labels + instance count.
        Padding labels (-1) are ignored.
        """
        valid_mask = labels != -1
        unique = np.unique(labels[valid_mask])

        if len(unique) == 0:
            # No valid instances: return all -1s and count = 0
            return np.full_like(labels, -1, dtype=np.int64), 0

        remap = {old: new for new, old in enumerate(unique)}
        relabeled = np.array([remap.get(l, -1) for l in labels], dtype=np.int64)
        return relabeled, len(unique)

    def pad_or_subsample(self, points, labels):
        """Ensures a fixed number of points per cloud using padding or subsampling."""
        num_points = points.shape[0]

        if num_points > self.max_points:
            # Downsample
            if self.sampling_method == 'fps':
                indices = self._farthest_point_sampling(points, self.max_points)
            else:
                # Randomly sample points
                indices = np.random.choice(num_points, self.max_points, replace=False)
            
            points, labels = points[indices], labels[indices]
            
        elif num_points < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - num_points
            pad_points = np.zeros((pad_size, 3), dtype=np.float64)
            pad_labels = np.full(pad_size, -1, dtype=np.int64)
            points = np.vstack((points, pad_points))
            labels = np.hstack((labels, pad_labels))

        return points, labels

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        point_path = os.path.join(self.point_folder, self.point_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        point_cloud = self.load_point_cloud(point_path)
        labels = self.load_labels(label_path, num_points=point_cloud.shape[0])

        # 1. Subsample/Pad
        point_cloud, labels = self.pad_or_subsample(point_cloud, labels)
        
        # 2. DATA AUGMENTATION (Training Only)
        # We apply this BEFORE Normal Calculation so normals are consistent with rotated geometry
        if self.mode == 'train' and AUGMENTATION_AVAILABLE:
            point_cloud = DataAugmentation.apply_rs(point_cloud)
        
        # 3. Compute Normals (on augmented geometry)
        if Config.input_dim == 6:
            point_cloud = compute_normals(point_cloud)

        # 4. Normalize
        point_cloud = normalize_pc(point_cloud)

        # 5. Relabel
        if self.mode == "train":
            labels, instance_count = self.relabel_instances(labels)
        else:
            instance_count = len(np.unique(labels[labels != -1]))

        # # Apply padding or subsampling
        # point_cloud, labels = self.pad_or_subsample(point_cloud, labels)
        
        # # --- METHODOLOGICAL INTEGRATION (Normals + Normalization) ---
        # # 1. Compute Normals if configured (Enhancement #1)
        # if Config.input_dim == 6:
        #     # Note: compute_normals expects (N, 3) and returns (N, 6)
        #     point_cloud = compute_normals(point_cloud)

        # # 2. Normalize XYZ (Essential for stability)
        # point_cloud = normalize_pc(point_cloud)
        # # ------------------------------------------------------------

        # # Relabel + get instance count (done last to handle padded -1s correctly)
        # if self.mode == "train":
        #     labels, instance_count = self.relabel_instances(labels)
        # else:
        #     instance_count = len(np.unique(labels[labels != -1]))

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), instance_count



# ==========================================
# 4. MODEL ARCHITECTURE (DGCNN + UNet)
# ==========================================
class DGCNNEncoder(nn.Module):
    """
    Innovative Encoder using Dynamic Graph CNN (Wang et al., 2019).
    dynamically computes graphs in feature space to capture local geometric structure.
    """
    def __init__(self, input_dim=6, k=20, emb_dim=128):
        super(DGCNNEncoder, self).__init__()
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for DGCNN.")
            
        self.k = k
        
        # EdgeConv 1
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(input_dim * 2, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2)), k=k)
        # EdgeConv 2
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(64 * 2, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2)), k=k)
        # EdgeConv 3
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(64 * 2, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2)), k=k)
        
        # Final projection
        self.lin_block = nn.Sequential(
            nn.Linear(128, emb_dim), nn.BatchNorm1d(emb_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        # x shape input: (Batch, Channels, Points)
        B, C, N = x.shape
        
        # Reshape for PyG: (Total_Points, Channels)
        x_flat = x.permute(0, 2, 1).contiguous().view(B * N, C)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)

        # Dynamic Edge Convolutions
        x1 = self.conv1(x_flat, batch_idx)
        x2 = self.conv2(x1, batch_idx)
        x3 = self.conv3(x2, batch_idx)
        
        # Local features
        local_feat = self.lin_block(x3) # (Total_Points, emb_dim)
        
        # Global Max Pool
        global_feat = global_max_pool(local_feat, batch_idx) # (B, emb_dim)
        
        # Reshape back to standard tensor format
        # Local: (B, N, emb_dim) -> (B, emb_dim, N)
        local_feat = local_feat.view(B, N, -1).permute(0, 2, 1)
        
        # Expand global to match local
        global_feat_expanded = global_feat.unsqueeze(-1).repeat(1, 1, N)
        
        # Concatenate for Skip Connection capability
        # Returns: (B, 2*emb_dim, N)
        return torch.cat([local_feat, global_feat_expanded], dim=1)


class SipUNetDecoder(nn.Module):
    """
    Decodes the global+local features back to per-point embeddings.
    Standard 1D Convolutional Decoder.
    """
    def __init__(self, input_channels, output_dim=64):
        super(SipUNetDecoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # No Activation on final layer (embedding space)
        return x


class PointUNet(nn.Module):
    """
    Main Model Wrapper.
    Encoder: DGCNN (Context Aware)
    Decoder: Simple Conv (Pointwise)
    """
    def __init__(self, config):
        super(PointUNet, self).__init__()
        self.encoder = DGCNNEncoder(input_dim=config.input_dim, k=config.k_neighbors, emb_dim=config.emb_dim)
        # Encoder outputs 2*emb_dim (local + global)
        self.decoder = SipUNetDecoder(input_channels=config.emb_dim * 2, output_dim=config.output_dim)

    def forward(self, x):
        # x: (B, N, C) -> Permute to (B, C, N)
        x = x.permute(0, 2, 1)
        features = self.encoder(x) # (B, 2*emb, N)
        embeddings = self.decoder(features) # (B, out_dim, N)
        return embeddings.permute(0, 2, 1) # Return (B, N, out_dim)


# ==========================================
# 5. DISCRIMINATIVE LOSS
# ==========================================
class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for instance segmentation in embedding space.
    (De Brabandere et al. 2017)
    
    Encourages:
        - points of the same instance to be close to their cluster center (delta_v)
        - instance centers to be far apart (delta_d)
        - small norm of cluster centers (regularizer)
    """
    def __init__(
        self, 
        delta_v: float = 0.3, 
        delta_d: float = 1.5, 
        alpha: float = 1.0, 
        beta: float = 1.0, 
        gamma: float = 0.001,
        ignore_label: int = -1
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_label = ignore_label

    def forward(self, embeddings, instance_labels):
        """
        Args:
            embeddings: (B, N, D)
            instance_labels: (B, N)
        """
        batch_size = embeddings.shape[0]
        total_loss = 0.0
        valid_batches = 0

        for b in range(batch_size):
            emb = embeddings[b] # (N, D)
            lbl = instance_labels[b] # (N,)
            
            # Filter ignore label
            mask = lbl != self.ignore_label
            if mask.sum() == 0:
                continue

            emb_valid = emb[mask]
            lbl_valid = lbl[mask]
            
            unique_labels = torch.unique(lbl_valid)
            num_instances = len(unique_labels)
            
            if num_instances == 0:
                continue

            # Compute cluster centers
            mu_list = []
            for inst_id in unique_labels:
                inst_mask = (lbl_valid == inst_id)
                mu_list.append(emb_valid[inst_mask].mean(dim=0))
            
            mu_tensor = torch.stack(mu_list) # (K, D)

            # --- 1. Variance Term (Pull) ---
            l_var = 0.0
            for i, inst_id in enumerate(unique_labels):
                inst_mask = (lbl_valid == inst_id)
                inst_emb = emb_valid[inst_mask]
                center = mu_tensor[i].unsqueeze(0) # (1, D)
                
                # Distance of points to their own center
                dist = torch.norm(inst_emb - center, dim=1)
                
                # Hinge: max(0, dist - delta_v)^2
                hinge = torch.clamp(dist - self.delta_v, min=0) ** 2
                l_var += hinge.mean()
            
            l_var /= num_instances
            
            # --- 2. Distance Term (Push) ---
            l_dist = 0.0
            if num_instances > 1:
                # Optimized: Use cdist instead of nested loops for performance
                # Using p=1 (Manhattan) to match provided snippet integration request
                # CHANGED to p=2 (Euclidean) to match Variance Term and HDBSCAN
                dist_mat = torch.cdist(mu_tensor, mu_tensor, p=2)
                
                # Create mask to ignore diagonal (dist to self is 0)
                diag_mask = torch.eye(num_instances, device=embeddings.device).bool()
                
                # Hinge: max(0, 2*delta_d - dist)^2
                dist_hinge = torch.clamp(2 * self.delta_d - dist_mat, min=0) ** 2
                
                # Zero out diagonals
                dist_hinge[diag_mask] = 0.0
                
                l_dist = dist_hinge.sum() / (num_instances * (num_instances - 1))
            
            # --- 3. Regularization Term ---
            # Using p=1 (L1 norm) for regularization as per snippet
            # CHANGED to p=2 (L2 norm) to be consistent
            l_reg = torch.norm(mu_tensor, p=2, dim=1).mean()

            # Combine
            total_loss += self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return total_loss / valid_batches




# ==========================================
# 6. CLUSTERING (INFERENCE)
# ==========================================
def cluster_embeddings(embeddings, method='meanshift'):
    """
    Post-processing: Turns embeddings into instance labels.
    Args:
        embeddings: (N, D) numpy array
        method: 'meanshift' or 'hdbscan' or 'dbscan'
    Returns:
        labels: (N,) numpy array
    """
    if method == 'meanshift':
        # Bandwidth is the crucial hyperparam here
        # Can be estimated via sklearn.cluster.estimate_bandwidth
        ms = MeanShift(bandwidth=0.6, bin_seeding=True)
        labels = ms.fit_predict(embeddings)
        
    elif method == 'dbscan':
        # More robust, faster than MeanShift for large N
        db = DBSCAN(eps=0.5, min_samples=10)
        labels = db.fit_predict(embeddings)
        
    elif method == 'hdbscan':
        if SKLEARN_HDBSCAN:
            # Robust to variable densities, no 'epsilon' parameter needed
            # min_cluster_size: Smallest valid roof face size
            # min_samples: Measure of 'conservativeness' (larger = more points marked as noise)
            clusterer = HDBSCAN(min_cluster_size=10, min_samples=5, cluster_selection_method='eom', cluster_selection_epsilon=0.1)
            labels = clusterer.fit_predict(embeddings)
        else:
            print("HDBSCAN requested but not installed. Falling back to DBSCAN.")
            db = DBSCAN(eps=0.5, min_samples=10)
            labels = db.fit_predict(embeddings)
        
    # TODO: Implement RANSAC plane fitting refinement here for production
    
    return labels


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def train_pipeline(conf: Config = None, data_root="data/roofNTNU/train_test_split", json_log_path="training_log.json", save_model_path="roof_segmentation_dgcnn_best.pth"):
    # 1. Setup
    if conf is None:
        conf = Config()
    
    if not PYG_AVAILABLE:
        print("Cannot run without PyG. Exiting.")
        return

    print(f"Initializing Model: DGCNN -> PointUNet on {conf.device}")
    model = PointUNet(conf).to(conf.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    
    # Initialize Loss Module
    criterion = DiscriminativeLoss(
        delta_v=conf.delta_v,
        delta_d=conf.delta_d,
        alpha=conf.alpha,
        beta=conf.beta,
        gamma=conf.gamma
    ).to(conf.device)
    
    # Prepare Logging
    # We serialize the config class attributes to a dict
    log_data = {
        "config": {k: str(v) for k, v in Config.__dict__.items() if not k.startswith('__')},
        "history": []
    }
    
    # 2. Data
    # root_dir should contain the 'train_test_split' folder
    # data_root = "data/roofNTNU/train_test_split" 
    
    # Initialize datasets for Train and Validation splits
    train_dataset = LiDARPointCloudDataset(
        base_dir=data_root, 
        split='train', 
        max_points=conf.max_points,
        sampling_method=conf.sampling_method
    )
    
    val_dataset = LiDARPointCloudDataset(
        base_dir=data_root, 
        split='val', 
        max_points=conf.max_points,
        sampling_method=conf.sampling_method
    )
    
    # Check if data exists
    if len(train_dataset) == 0:
        print(f"No training files found in '{data_root}/'. Exiting.")
        # return # Commented out for dry-run/template purposes

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=conf.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=conf.batch_size, 
        shuffle=False,
        pin_memory=True
    )
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(conf.epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{conf.epochs} [Train]")
        
        for points, labels, _ in loop:
            points = points.to(conf.device)
            labels = labels.to(conf.device)
            
            optimizer.zero_grad()
            embeddings = model(points)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        if len(val_loader) > 0:
            with torch.no_grad():
                for points, labels, _ in val_loader:
                    points = points.to(conf.device)
                    labels = labels.to(conf.device)
                    
                    embeddings = model(points)
                    loss = criterion(embeddings, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "roof_segmentation_dgcnn_best.pth")
                print("  -> New best model saved.")
        else:
            avg_val_loss = None
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} (No Validation Data)")
        
        # --- LOGGING ---
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }
        log_data["history"].append(epoch_stats)
        
        # Save log to JSON file (overwriting each epoch to keep it current)
        with open(json_log_path, "w") as f:
            json.dump(log_data, f, indent=4)
        
        # 4. Quick Inference Check (On Validation Sample)
        if epoch % 10 == 0 and len(val_dataset) > 0:
            with torch.no_grad():
                # Grab a sample from validation set
                points_sample, _, _ = val_dataset[0]
                points_sample = points_sample.unsqueeze(0).to(conf.device)
                
                sample_emb = model(points_sample)[0].cpu().numpy() # (N, D)
                pred_labels = cluster_embeddings(sample_emb, method='hdbscan')
                print(f"  [Inference Check] Found {len(np.unique(pred_labels))} instances in a validation sample.")

    # Save final model
    torch.save(model.state_dict(), save_model_path)
    print("Training Complete. Log saved to training_log.json.")


if __name__ == "__main__":
    conf = Config()
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("models"):
        os.makedirs("models")
    log_path = os.path.join("logs", "training_log_20251209_da_rs.json")
    model_path = os.path.join("models", "roof_segmentation_dgcnn_20251209_da_rs.pth")
    train_pipeline(conf, data_root="data/roofNTNU/train_test_split", json_log_path=log_path, save_model_path=model_path)
# ==========================================
# End of Script



