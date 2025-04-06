# data/BrainMRIDataset_Class.py

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage.transform import resize

class BrainMRIDataset(Dataset):
    """PyTorch Dataset for loading BraTS2020 data from local file system.
    
    Attributes:
        root_dir (str): Root directory containing patient folders
        img_size (int): Target size for image resizing
        slice_cache (list): Cache of preprocessed (image, mask) tuples
        target_shape (tuple): Target spatial dimensions (img_size, img_size)
    """
    
    def __init__(self, root_dir='BraTS2020_TrainingData', img_size=160):
        """Initialize dataset with local file paths.
        
        Args:
            root_dir: Path to MICCAI_BraTS2020_TrainingData folder
            img_size: Target size for resizing slices
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.target_shape = (img_size, img_size)
        
        # Validate directory structure
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root directory {self.root_dir} not found")
            
        # Get list of patient directories
        self.patient_dirs = self._get_patient_paths()
        print(f"Found {len(self.patient_dirs)} patient directories")
        
        # Cache for preprocessed slices
        self.slice_cache = []
        self._build_memory_efficient_cache()

    def _get_patient_paths(self):
        """Get sorted list of patient directory paths.
        
        Returns:
            list: Sorted list of full paths to patient directories
        """
        patients = [d for d in os.listdir(self.root_dir) 
                  if os.path.isdir(os.path.join(self.root_dir, d))]
        return sorted([os.path.join(self.root_dir, p) for p in patients])

    def _load_nifti(self, file_path):
        """Load NIfTI file and return its data array.
        
        Args:
            file_path: Path to .nii or .nii.gz file
            
        Returns:
            np.ndarray: 3D volume data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NIfTI file {file_path} not found")
        return nib.load(file_path).get_fdata()

    def _build_memory_efficient_cache(self):
        """Preprocess and cache only tumor-containing slices."""
        for patient_path in tqdm(self.patient_dirs, desc='Processing patients'):
            try:
                # Load segmentation mask
                seg_path = os.path.join(patient_path, 
                    f"{os.path.basename(patient_path)}_seg.nii")
                seg = self._load_nifti(seg_path)

                # Load all modalities
                modalities = {}
                for mod in ['flair', 't1', 't1ce', 't2']:
                    mod_path = os.path.join(patient_path,
                        f"{os.path.basename(patient_path)}_{mod}.nii")
                    modalities[mod] = self._load_nifti(mod_path)

                # Print volume shapes for first patient
                if not hasattr(self, 'shape_info_printed'):
                    print("\nOriginal 3D Volume Shapes:")
                    print(f"Segmentation: {seg.shape}")
                    for mod, data in modalities.items():
                        print(f"{mod.upper()}: {data.shape}")
                    self.shape_info_printed = True

                # Process tumor-containing slices
                for z in range(seg.shape[2]):
                    if np.any(seg[:, :, z] > 0):
                        # Stack and preprocess modalities
                        img_slices = [self._process_modality(modalities[mod][..., z]) 
                                    for mod in ['flair', 't1', 't1ce', 't2']]
                        img = np.stack(img_slices, axis=-1)
                        img = torch.tensor(img).permute(2, 0, 1).float()

                        # Process and remap mask
                        mask = seg[..., z].copy()
                        mask[mask == 4] = 3  # Merge labels 3 and 4
                        mask = self._resize_mask(mask)
                        mask = torch.tensor(mask).long()

                        self.slice_cache.append((img, mask))

            except Exception as e:
                print(f"Error processing {patient_path}: {str(e)}")
                continue

    def _process_modality(self, img_slice):
        """Preprocess a single modality slice.
        
        Args:
            img_slice: 2D numpy array from NIfTI slice
            
        Returns:
            np.ndarray: Resized and normalized slice
        """
        img_resized = resize(img_slice, self.target_shape, 
                           mode='constant', preserve_range=True,
                           anti_aliasing=True)
        # Normalize to [0, 1]
        return (img_resized / (img_resized.max() + 1e-7)).astype(np.float32)

    def _resize_mask(self, mask):
        """Resize mask with nearest-neighbor interpolation.
        
        Args:
            mask: 2D numpy array of segmentation labels
            
        Returns:
            np.ndarray: Resized mask with original labels
        """
        return resize(mask, self.target_shape, order=0, 
                     mode='edge', preserve_range=True).astype(np.uint8)

    def __len__(self):
        """Return number of cached slices."""
        return len(self.slice_cache)

    def __getitem__(self, idx):
        """Return (image, mask) tuple at given index.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            tuple: (4-channel image tensor, mask tensor)
        """
        img, mask = self.slice_cache[idx]
        return img.clone(), mask.clone()