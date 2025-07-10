"""Custom Dataset and Sampler for BraTS 2020 MRI data.

This module provides an optimized PyTorch Dataset class for loading and
preprocessing 3D MRI scans from the BraTS 2020 dataset. It also includes a
custom Sampler to improve data loading efficiency by grouping slices by patient.
"""

import os
import glob
import random

import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Sampler

class BrainMRIDataset_Optimized(Dataset):
    
    """An optimized Dataset for loading BraTS 2020 MRI slices.

    This class loads 2D slices from 3D .nii.gz MRI scans. It is optimized for
    speed by pre-computing normalization statistics and using a patient-level
    cache to avoid repeatedly loading the same 3D volume from disk. It handles
    four modalities (FLAIR, T1, T1ce, T2) and the segmentation mask.

    Attributes:
        root_dir (str): Path to the root directory containing patient folders.
        img_size (int): The target size to resize 2D slices to.
        patient_list (list): A list of patient IDs to include in the dataset.
        negative_slice_ratio (float): The ratio of slices without tumors to include.
        transforms (albumentations.Compose): Albumentations transforms for data augmentation.
    """
    
    def __init__(
        self,
        root_dir,
        img_size=224,
        patient_list=None,
        negative_slice_ratio=0.25,
        transforms=None
    ):
        """Initializes the dataset object."""
        
        self.root_dir = root_dir
        self.img_size = img_size
        self.target_shape = (img_size, img_size)
        self.negative_slice_ratio = negative_slice_ratio
        self.transforms = transforms
        
        self.patient_dirs = sorted(patient_list) if patient_list is not None else []
        print(f"Initializing dataset with {len(self.patient_dirs)} patient directories.")
        if self.transforms:
            print("Data augmentation will be applied on-the-fly.")

        self.patient_filepaths = self._get_patient_filepaths()
        
        # --- NEW: Pre-compute normalization stats for all patients ---
        self.norm_stats = self._precompute_normalization_stats()
        
        self.slice_pointers = self._build_slice_pointers()
        self.patient_cache = {'id': None, 'data': None, 'seg': None}
        
        print(f"Dataset ready. Total slices to be loaded on-the-fly: {len(self.slice_pointers)}")

    def _get_patient_filepaths(self):
        """Scans the root directory to find file paths for each modality for each patient."""
        
        patient_filepaths = {}
        valid_patient_dirs = [pid for pid in self.patient_dirs if os.path.isdir(os.path.join(self.root_dir, pid))]
        for patient_id in tqdm(valid_patient_dirs, desc="Finding file paths"):
            patient_folder_path = os.path.join(self.root_dir, patient_id)
            modalities = ['flair', 't1', 't1ce', 't2', 'seg']
            paths = {}
            all_found = True
            for mod in modalities:
                pattern = os.path.join(patient_folder_path, f"{patient_id}_{mod}.nii*")
                files_found = glob.glob(pattern)
                if not files_found:
                    print(f"Warning: Missing '{mod}' file for patient {patient_id}. Skipping patient.")
                    all_found = False
                    break
                paths[mod] = files_found[0]
            if all_found:
                patient_filepaths[patient_id] = paths
        return patient_filepaths
    
    def _precompute_normalization_stats(self):
        """Calculates and stores the mean and std for each patient's 3D volume."""
        
        stats = {}
        for patient_id, paths in tqdm(self.patient_filepaths.items(), desc="Pre-computing norm stats"):
            stats[patient_id] = {}
            for mod in ['flair', 't1', 't1ce', 't2']:
                img = nib.load(paths[mod]).get_fdata(dtype=np.float32)
                non_zero_voxels = img[np.nonzero(img)]
                if non_zero_voxels.size > 0:
                    stats[patient_id][mod] = {
                        'mean': non_zero_voxels.mean(),
                        'std': non_zero_voxels.std()
                    }
                else:
                    stats[patient_id][mod] = {'mean': 0, 'std': 1}
        return stats

    def _build_slice_pointers(self):
        """Creates a list of (patient_id, slice_index) tuples to be loaded."""
        
        slice_pointers = []
        for patient_id in tqdm(self.patient_filepaths.keys(), desc="Building slice pointers"):
            paths = self.patient_filepaths[patient_id]
            
            seg_vol   = nib.load(paths['seg']).get_fdata().astype(np.uint8)
            flair_vol = nib.load(paths['flair']).get_fdata(dtype=np.float32)

            for z in range(seg_vol.shape[2]):
                mask_slice = seg_vol[..., z]

                # Check for signal using only the flair slice. This is much faster.
                has_signal = flair_vol[..., z].any()

                # Keep the rest of your logic the same
                if has_signal and (
                    mask_slice.any()
                    or (np.random.rand() < self.negative_slice_ratio)
                ):
                    slice_pointers.append((patient_id, z))

        return slice_pointers



    def _load_and_cache_patient(self, patient_id):
        """Loads a patient's full 3D data into a cache for quick access."""
        
        filepaths = self.patient_filepaths[patient_id]
        modalities_data = {}
        
        for mod in ['flair', 't1', 't1ce', 't2']:
            mod_img = nib.load(filepaths[mod])
            mod_data = mod_img.get_fdata(dtype=np.float32)
            
            # --- MODIFIED: Use pre-computed stats for normalization ---
            stats = self.norm_stats[patient_id][mod]
            mean, std = stats['mean'], stats['std']
            
            mod_data = (mod_data - mean) / (std + 1e-8)
            mod_data = np.clip(mod_data, -5.0, 5.0)
            modalities_data[mod] = mod_data

        seg_img = nib.load(filepaths['seg'])
        seg_data = seg_img.get_fdata().astype(np.uint8)
        seg_data[seg_data == 4] = 3

        self.patient_cache['id'] = patient_id
        self.patient_cache['data'] = modalities_data
        self.patient_cache['seg'] = seg_data

    def __len__(self):
        """Returns the total number of slices in the dataset."""
        
        return len(self.slice_pointers)

    def __getitem__(self, idx):
        """Retrieves and processes a single 2D slice and its mask."""
        
        patient_id, slice_idx = self.slice_pointers[idx]

        if self.patient_cache.get('id') != patient_id:
            self._load_and_cache_patient(patient_id)
        
        modalities_data_3d = self.patient_cache['data']
        seg_data_3d = self.patient_cache['seg']

        img_slices = [modalities_data_3d[mod][..., slice_idx] for mod in ['flair', 't1', 't1ce', 't2']]
        img_np = np.stack(img_slices, axis=-1)
        mask_np = seg_data_3d[..., slice_idx]

        # --- MODIFIED: Use fast OpenCV resizing ---
        # Use bilinear interpolation for the image (fast and good quality)
        img_np = cv2.resize(img_np, self.target_shape, interpolation=cv2.INTER_LINEAR)
        # Use nearest-neighbor interpolation for the mask to preserve label integrity
        mask_np = cv2.resize(mask_np, self.target_shape, interpolation=cv2.INTER_NEAREST)

        # Ensure image is back to (H, W, C) if cv2 flattens it
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        if self.transforms:
            augmented = self.transforms(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor
    
    
    
class PatientSampler(Sampler):
    
    """A custom sampler that groups slices by patient to improve cache efficiency.

    This sampler shuffles the order of patients and then yields all slices
    from one patient before moving to the next. This strategy ensures that
    the `_load_and_cache_patient` method in the Dataset class is called as
    infrequently as possible, significantly speeding up epoch times.

    Args:
        slice_pointers (list): The list of (patient_id, slice_index) pointers
                               from the `BrainMRIDataset_Optimized` instance.
    """
    
    def __init__(self, slice_pointers):
        """Initializes the PatientSampler."""
        
        self.slice_pointers = slice_pointers
        
        # Create a mapping from patient_id to a list of slice indices
        self.patient_to_indices = {}
        for idx, (patient_id, slice_idx) in enumerate(self.slice_pointers):
            if patient_id not in self.patient_to_indices:
                self.patient_to_indices[patient_id] = []
            self.patient_to_indices[patient_id].append(idx)
            
        self.patient_ids = list(self.patient_to_indices.keys())
        self.num_samples = len(self.slice_pointers)

    def __iter__(self):
        """Yields a shuffled sequence of slice indices, grouped by patient."""
        
        # Shuffle patient order
        random.shuffle(self.patient_ids)
        
        # Iterate through shuffled patients
        for patient_id in self.patient_ids:
            indices = self.patient_to_indices[patient_id]
            # Shuffle slice order within the patient
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self):
        """Returns the total number of samples."""
        
        return self.num_samples