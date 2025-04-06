import unittest
import os
import shutil
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent.parent))
from data.brain_mri_dataset import BrainMRIDataset

class TestBrainMRIDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data structure once before all tests."""
        # Create test directory structure
        cls.test_dir = 'test_brats_data'
        cls.training_data_dir = os.path.join(cls.test_dir, 'MICCAI_BraTS2020_TrainingData')
        os.makedirs(cls.training_data_dir, exist_ok=True)
        
        # Create two mock patient directories
        cls.patient_dirs = []
        for i in range(1, 3):
            patient_id = f'BraTS20_Training_{i:03d}'
            patient_path = os.path.join(cls.training_data_dir, patient_id)
            os.makedirs(patient_path, exist_ok=True)
            cls.patient_dirs.append(patient_path)
            
            # Create mock data for each patient
            cls._create_mock_data(patient_path, patient_id)
    
    @classmethod
    def _create_mock_data(cls, patient_path, patient_id):
        """Create mock MRI and segmentation data for testing."""
        # Create a small 3D volume (20x20x20)
        vol_shape = (20, 20, 20)
        
        # Create mock segmentation with some tumor regions
        seg_data = np.zeros(vol_shape, dtype=np.uint8)
        # Add some tumor in slices 8-12
        seg_data[5:15, 5:15, 8:12] = 1  # Tumor core
        seg_data[7:13, 7:13, 9:11] = 2  # Enhancing tumor
        seg_data[3:17, 3:17, 7:13] = 4  # Edema (will be remapped to 3)
        
        # Create mock modality data
        modalities = {
            'flair': np.random.rand(*vol_shape) * 1000,
            't1': np.random.rand(*vol_shape) * 1000,
            't1ce': np.random.rand(*vol_shape) * 1000,
            't2': np.random.rand(*vol_shape) * 1000
        }
        
        # Save as NIfTI files
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(seg_data, affine), os.path.join(patient_path, f"{patient_id}_seg.nii.gz"))
        
        for mod_name, mod_data in modalities.items():
            nib.save(nib.Nifti1Image(mod_data, affine), 
                     os.path.join(patient_path, f"{patient_id}_{mod_name}.nii.gz"))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data after all tests have run."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir, img_size=128)
        self.assertEqual(len(dataset.patient_dirs), 2)
        self.assertEqual(dataset.img_size, 128)
        self.assertEqual(dataset.target_shape, (128, 128))
    
    def test_list_patient_dirs(self):
        """Test patient directory listing functionality."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        patient_dirs = dataset._list_patient_dirs()
        self.assertEqual(len(patient_dirs), 2)
        for dir_path in patient_dirs:
            self.assertTrue(os.path.basename(dir_path).startswith('BraTS20_Training_'))
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directories."""
        with self.assertRaises(FileNotFoundError):
            BrainMRIDataset(dataset_path='nonexistent_directory')
    
    def test_cache_building(self):
        """Test that the slice cache is built correctly."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        # We should have some slices with tumors in the cache
        self.assertGreater(len(dataset.slice_cache), 0)
    
    def test_getitem(self):
        """Test the __getitem__ method."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        if len(dataset) > 0:
            img, mask = dataset[0]
            # Check tensor shapes
            self.assertEqual(img.shape[0], 4)  # 4 modalities
            self.assertEqual(img.shape[1], dataset.img_size)
            self.assertEqual(img.shape[2], dataset.img_size)
            self.assertEqual(mask.shape[0], dataset.img_size)
            self.assertEqual(mask.shape[1], dataset.img_size)
            # Check tensor types
            self.assertEqual(img.dtype, torch.float32)
            self.assertEqual(mask.dtype, torch.int64)
    
    def test_modality_processing(self):
        """Test the modality processing method."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        # Create a test slice
        test_slice = np.random.rand(30, 30) * 100
        processed = dataset._process_modality(test_slice)
        # Check shape and normalization
        self.assertEqual(processed.shape, dataset.target_shape)
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
        self.assertEqual(processed.dtype, np.float32)
    
    def test_mask_processing(self):
        """Test the mask processing method."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        # Create a test mask with different labels
        test_mask = np.zeros((30, 30), dtype=np.uint8)
        test_mask[10:20, 10:20] = 1
        test_mask[12:18, 12:18] = 2
        test_mask[15:25, 15:25] = 4  # Should be remapped to 3
        
        processed = dataset._resize_mask(test_mask)
        # Check shape and label preservation
        self.assertEqual(processed.shape, dataset.target_shape)
        self.assertEqual(processed.dtype, np.uint8)
        # Values 0, 1, 2 should be preserved
        self.assertTrue(0 in processed)
        self.assertTrue(1 in processed)
        self.assertTrue(2 in processed)
    
    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Try getting a batch
        if len(dataset) >= 2:
            batch = next(iter(loader))
            images, masks = batch
            
            # Check batch shapes
            self.assertEqual(images.shape[0], min(2, len(dataset)))  # Batch size
            self.assertEqual(images.shape[1], 4)  # 4 modalities
            self.assertEqual(images.shape[2], dataset.img_size)
            self.assertEqual(images.shape[3], dataset.img_size)
            
            self.assertEqual(masks.shape[0], min(2, len(dataset)))  # Batch size
            self.assertEqual(masks.shape[1], dataset.img_size)
            self.assertEqual(masks.shape[2], dataset.img_size)
    
    def test_len(self):
        """Test the __len__ method."""
        dataset = BrainMRIDataset(dataset_path=self.test_dir)
        self.assertEqual(len(dataset), len(dataset.slice_cache))

if __name__ == '__main__':
    unittest.main()