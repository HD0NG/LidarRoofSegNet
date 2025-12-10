import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor

class RoofPostProcessor:
    """
    Post-processing tools to refine roof segmentation boundaries (hinges).
    """

    @staticmethod
    def knn_smoothing(points, labels, k=10):
        """
        Method 1: Majority Voting.
        Replaces a point's label with the most common label among its k-nearest neighbors.
        Smoothes out jagged edges.
        """
        N = points.shape[0]
        refined_labels = labels.copy()
        
        # Build neighbor graph using XYZ only
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points[:, :3])
        indices = nbrs.kneighbors(points[:, :3], return_distance=False)
        
        # Iterate and vote
        for i in range(N):
            # Get labels of neighbors (excluding self at index 0)
            neighbor_indices = indices[i, 1:]
            neighbor_labels = labels[neighbor_indices]
            
            # Remove noise labels (-1) from voting if possible
            valid_votes = neighbor_labels[neighbor_labels != -1]
            
            if len(valid_votes) > 0:
                # Find mode
                vals, counts = np.unique(valid_votes, return_counts=True)
                majority_label = vals[np.argmax(counts)]
                refined_labels[i] = majority_label
                
        return refined_labels

    @staticmethod
    def ransac_refinement(points, labels, residual_threshold=0.1):
        """
        Method 2: RANSAC Plane Fitting.
        1. Fits a geometric plane to each cluster.
        2. Re-assigns points to the plane they fit best.
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1] # Ignore noise
        
        plane_models = {}
        
        # 1. Fit Planes
        for lbl in unique_labels:
            mask = (labels == lbl)
            pts_cluster = points[mask]
            
            if len(pts_cluster) < 10: continue # Skip tiny clusters
            
            # Fit Plane: Z = aX + bY + d
            ransac = RANSACRegressor(residual_threshold=residual_threshold)
            try:
                ransac.fit(pts_cluster[:, :2], pts_cluster[:, 2]) # Predict Z from X,Y
                plane_models[lbl] = ransac
            except:
                continue

        # 2. Re-assign boundary points
        refined_labels = labels.copy()
        
        # Check every point against every plane
        for i in range(len(points)):
            current_lbl = labels[i]
            
            # Only checking points that are already assigned (or you could check noise too)
            if current_lbl == -1: continue 
            
            p = points[i:i+1] # (1, 3)
            best_dist = float('inf')
            best_lbl = current_lbl
            
            # Calculate distance to current plane
            if current_lbl in plane_models:
                pred_z = plane_models[current_lbl].predict(p[:, :2])
                dist = abs(pred_z - p[:, 2])
                best_dist = dist[0]
            
            # Check if it fits better on another plane
            for lbl, model in plane_models.items():
                if lbl == current_lbl: continue
                
                pred_z = model.predict(p[:, :2])
                dist = abs(pred_z - p[:, 2])[0]
                
                # If significantly closer to another plane (and physically close)
                # We add a 'margin' to prevent flickering
                if dist < best_dist - 0.05: 
                    best_dist = dist
                    best_lbl = lbl
            
            refined_labels[i] = best_lbl
            
        return refined_labels

    @staticmethod
    def recover_noise_points(points, labels, k=5, normal_threshold=0.8):
        """
        Method 3: Unassigned Point Recovery.
        Assigns points labeled as noise (-1) to the nearest valid plane cluster.
        If input 'points' has 6 columns (XYZ + Normals), it enforces geometric 
        consistency (dot product > normal_threshold).
        """
        refined_labels = labels.copy()
        
        # 1. Identify Noise vs Valid
        noise_mask = (labels == -1)
        valid_mask = (labels != -1)
        
        if not np.any(noise_mask) or not np.any(valid_mask):
            return refined_labels # Nothing to recover or nothing to assign to
            
        noise_indices = np.where(noise_mask)[0]
        valid_points = points[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Check if we have normals (Shape N, 6)
        has_normals = points.shape[1] >= 6
        
        # 2. Build Tree on Valid Points (Spatial only)
        # Using KDTree for fast spatial lookup
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(valid_points[:, :3])
        
        # Find k nearest valid neighbors for each noise point
        # dists: (N_noise, k), indices: (N_noise, k) (indices into valid_points array)
        noise_xyz = points[noise_mask][:, :3]
        dists, neighbor_indices = nbrs.kneighbors(noise_xyz)
        
        # 3. Assign
        for i, original_idx in enumerate(noise_indices):
            # Potential labels from neighbors
            potential_labels = valid_labels[neighbor_indices[i]]
            
            best_label = -1
            
            if has_normals:
                # Enforce Normal Consistency
                noise_normal = points[original_idx, 3:6]
                
                for j in range(k):
                    candidate_label = potential_labels[j]
                    idx_in_valid = neighbor_indices[i, j]
                    candidate_normal = valid_points[idx_in_valid, 3:6]
                    
                    # Dot product check: are faces pointing in similar direction?
                    # >0.8 is roughly <35 degrees difference
                    if np.dot(noise_normal, candidate_normal) > normal_threshold:
                        best_label = candidate_label
                        break
                
                # Fallback: if no normal matched, maybe stick with closest spatial if lenient?
                # For now, strict: if no normal match, stay as noise.
                if best_label == -1:
                    # Optional: relax constraint or take closest spatial
                    # best_label = potential_labels[0] 
                    pass
            else:
                # Spatial only: Take the closest neighbor's label
                best_label = potential_labels[0]
                
            refined_labels[original_idx] = best_label
            
        return refined_labels

# -----------------------------------------------------
# Usage Example
# -----------------------------------------------------
if __name__ == "__main__":
    # Mock Data (100 points, 3 dims)
    points = np.random.rand(100, 3)
    labels = np.random.randint(-1, 2, 100) # Includes -1 (noise)
    
    print(f"Original Noise Count: {np.sum(labels == -1)}")
    
    # 1. Smooth Jagged Edges
    smooth_labels = RoofPostProcessor.knn_smoothing(points, labels, k=10)
    
    # 2. Recover Noise (-1) Points
    recovered_labels = RoofPostProcessor.recover_noise_points(points, smooth_labels, k=5)
    print(f"Recovered Noise Count: {np.sum(recovered_labels == -1)}")
    
    # 3. Snap to Geometry
    geom_labels = RoofPostProcessor.ransac_refinement(points, recovered_labels, residual_threshold=0.05)