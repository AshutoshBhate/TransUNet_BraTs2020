import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm


class BrainMRIDataset(Dataset):
    """Dataset class for loading brain MRI data from BraTS2020.
    
    This class handles loading, preprocessing, and caching brain MRI data for
    tumor segmentation tasks. It efficiently processes data by only loading tumor-
    containing slices from 3D volumes, and standardizes all data to a fixed size.
    
    Attributes:
        dataset_path: Path to the BraTS2020_TrainingData directory.
        img_size: Target size for the processed images (square).
        target_shape: Tuple representing target dimensions for resizing.
        patient_dirs: List of patient directories found in the dataset.
        slice_cache: List of preprocessed (image, mask) tensor pairs.
    """

    def __init__(self, dataset_path='BraTS2020_TrainingData', img_size=160):
        """Initializes the BrainMRIDataset.
        
        Args:
            dataset_path: Path to the BraTS2020_TrainingData directory.
                Default: 'BraTS2020_TrainingData'
            img_size: Target size for the images (creates img_size x img_size images).
                Default: 160
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.target_shape = (img_size, img_size)
        
        # List all patient folders in the BraTS2020 dataset
        self.patient_dirs = self._list_patient_dirs()
        print(f"Found {len(self.patient_dirs)} patient directories in {dataset_path}.")
        
        # Cache for preprocessed slices
        self.slice_cache = []  # Format: [(img_tensor, mask_tensor), ...]
        self._build_memory_efficient_cache()

    def _list_patient_dirs(self):
        """Lists all patient directories in the BraTS2020 dataset folder.
        
        Returns:
            A sorted list of patient directory paths.
            
        Raises:
            FileNotFoundError: If the dataset directory doesn't exist.
        """
        # Check if main directory exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_path}' not found")
        
        # Path to the training data
        training_path = os.path.join(self.dataset_path, 'MICCAI_BraTS2020_TrainingData')
        if not os.path.exists(training_path):
            raise FileNotFoundError(f"Training data directory '{training_path}' not found")
            
        # Get all BraTS20_Training_XXX directories
        patient_dirs = []
        for item in os.listdir(training_path):
            if item.startswith("BraTS20_Training_"):
                item_path = os.path.join(training_path, item)
                if os.path.isdir(item_path):
                    patient_dirs.append(item_path)
                
        return sorted(patient_dirs)

    def _build_memory_efficient_cache(self):
        """Builds a cache of preprocessed image slices.
        
        Processes all patients' MRI data by:
        1. Loading segmentation masks and four MRI modalities (FLAIR, T1, T1CE, T2)
        2. Preprocessing each slice containing tumor regions
        3. Storing processed image/mask pairs in memory for efficient access
        
        Only slices containing tumor regions (non-zero values in segmentation mask)
        are processed to optimize memory usage.
        """
        for patient_path in tqdm(self.patient_dirs, desc='Building slice cache'):
            try:
                # Get patient ID from the folder name
                patient_id = os.path.basename(patient_path)
                
                # Find segmentation file
                seg_file = os.path.join(patient_path, f"{patient_id}_seg.nii")
                if not os.path.exists(seg_file):
                    seg_file = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
                    if not os.path.exists(seg_file):
                        print(f"⚠️ No segmentation file found for {patient_id}, skipping...")
                        continue
                
                # Load segmentation
                seg_img = nib.load(seg_file)
                seg = seg_img.get_fdata().copy()
                del seg_img  # Free memory
                
                # Load modalities
                modalities = {}
                for mod in ['flair', 't1', 't1ce', 't2']:
                    # Try both .nii and .nii.gz extensions
                    mod_file = os.path.join(patient_path, f"{patient_id}_{mod}.nii")
                    if not os.path.exists(mod_file):
                        mod_file = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
                        if not os.path.exists(mod_file):
                            print(f"⚠️ {mod} file missing for {patient_id}, skipping...")
                            continue
                    
                    # Load modality
                    mod_img = nib.load(mod_file)
                    modalities[mod] = mod_img.get_fdata().copy()
                    del mod_img  # Free memory
                
                if len(modalities) < 4:
                    print(f"⚠️ Missing modalities for {patient_id}, skipping...")
                    continue
                
                # After loading modalities for first patient, print shape information
                if not hasattr(self, 'shape_info_printed'):
                    print("\nOriginal 3D Volume Shapes:")
                    print(f"Segmentation: {seg.shape}")
                    for mod, data in modalities.items():
                        print(f"{mod.upper()}: {data.shape}")
                    print(f"Input type: {type(data)}")
                    self.shape_info_printed = True
                
                # Process only slices with tumor presence
                for z in range(seg.shape[2]):
                    if np.any(seg[:, :, z] > 0):  # Keep only tumor-containing slices
                        img_slices = [self._process_modality(modalities[mod][..., z]) 
                                      for mod in modalities]
                        img = np.stack(img_slices, axis=-1)  # Stack into multi-channel image
                        img = torch.tensor(img).permute(2, 0, 1).float()
                        
                        # Process mask
                        mask = seg[..., z].copy()
                        
                        # Print remapping info once
                        if not hasattr(self, 'remap_info_printed') and np.any(mask == 4):
                            print(f"\nUnique mask values BEFORE remapping: {np.unique(mask)}")
                            mask[mask == 4] = 3
                            print(f"Unique mask values AFTER remapping: {np.unique(mask)}")
                            self.remap_info_printed = True
                        else:
                            mask[mask == 4] = 3  # Always remap without printing
                        
                        mask = self._resize_mask(mask)
                        mask = torch.tensor(mask).long()
                        
                        self.slice_cache.append((img, mask))
                
                del modalities, seg  # Free memory
                
            except Exception as e:
                print(f"🚨 Error processing {os.path.basename(patient_path)}: {e}")
                continue

    def _process_modality(self, img_slice):
        """Processes a single modality slice.
        
        Resizes the slice to the target shape and normalizes values to [0,1].
        
        Args:
            img_slice: A 2D numpy array representing one slice of an MRI modality.
            
        Returns:
            A processed 2D numpy array of type float32.
        """
        # Print resizing info once
        if not hasattr(self, 'resize_info_printed'):
            original_shape = img_slice.shape
            print(f"\nResizing image slices from original {original_shape} to {self.target_shape}")
            self.resize_info_printed = True
            
        img_resized = resize(img_slice, self.target_shape, mode='constant', 
                             preserve_range=True, anti_aliasing=True)
        max_val = img_resized.max()
        if max_val > 0:
            img_resized /= (max_val + 1e-7)
        return img_resized.astype(np.float32)

    def _resize_mask(self, mask):
        """Resizes a segmentation mask.
        
        Uses nearest-neighbor interpolation to preserve label values.
        
        Args:
            mask: A 2D numpy array containing segmentation labels.
            
        Returns:
            A resized 2D numpy array of type uint8.
        """
        return resize(mask, self.target_shape, order=0, mode='edge',
                      preserve_range=True, anti_aliasing=False).astype(np.uint8)

    def __len__(self):
        """Returns the size of the dataset.
        
        Returns:clea
            The number of preprocessed slices in the cache.
        """
        return len(self.slice_cache)

    def __getitem__(self, idx):
        """Retrieves a specific item from the dataset.
        
        Args:
            idx: Index of the item to retrieve.
            
        Returns:
            A tuple of (image, mask) tensors, where image has shape [C,H,W]
            and mask has shape [H,W].
        """
        # Return a clone to prevent in-place modifications.
        img, mask = self.slice_cache[idx]
        return img.clone(), mask.clone()