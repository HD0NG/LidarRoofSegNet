import numpy as np

class DataAugmentation:
    """
    Standard geometric augmentations for Point Clouds.
    Apply these in __getitem__ during TRAINING only.
    """

    @staticmethod
    def random_rotate_z(points):
        """
        Rotates the point cloud around the Z-axis (up-axis).
        Roofs orientation shouldn't matter for segmentation.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ])
        
        # Apply to XYZ (first 3 cols)
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        # If we have normals (cols 3-6), rotate them too!
        if points.shape[1] >= 6:
            points[:, 3:6] = points[:, 3:6] @ rotation_matrix.T
            
        return points

    @staticmethod
    def random_jitter(points, sigma=0.01, clip=0.05):
        """
        Adds small Gaussian noise to point coordinates.
        Helps simulate sensor noise and prevents overfitting to exact positions.
        """
        N, C = points.shape
        # Only jitter XYZ, not normals/features
        jitter = np.clip(sigma * np.random.randn(N, 3), -clip, clip)
        points[:, :3] += jitter
        return points

    @staticmethod
    def random_scale(points, scale_low=0.8, scale_high=1.2):
        """
        Slightly scales the roof size. 
        Helps the model handle roofs of different sizes.
        """
        scale = np.random.uniform(scale_low, scale_high)
        points[:, :3] *= scale
        return points

    @staticmethod
    def apply_all(points):
        """Apply rotation, scaling, and jitter."""
        points = DataAugmentation.random_rotate_z(points)
        points = DataAugmentation.random_scale(points)
        points = DataAugmentation.random_jitter(points)
        return points
    
    @staticmethod
    def apply_rs(points):
        """Apply rotation, scaling, and jitter."""
        points = DataAugmentation.random_rotate_z(points)
        points = DataAugmentation.random_scale(points)
        # points = DataAugmentation.random_jitter(points)
        return points