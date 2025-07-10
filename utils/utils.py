"""Utility functions for the segmentation project.

This module contains helper functions for data augmentation, metric calculation,
model evaluation, and result visualization.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast
import medpy.metric as medpy_metric
import albumentations as A

from data.BrainMRIDataset_Class import BrainMRIDataset_Optimized

import matplotlib.pyplot as plt

def get_train_transforms(img_size):
    """Defines and returns the data augmentation pipeline for training.

    Args:
        img_size (int): The target image size. Not directly used in this
            pipeline but kept for API consistency.

    Returns:
        albumentations.Compose: A composition of augmentation transforms.
    """
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.6,
            border_mode=0      
        ),
        A.GridDistortion(p=0.2),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.2
        ),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5), # Changed from GaussSmooth for clarity, functionality is similar
        ], p=0.2),
        
        # CORRECTED PART: Keep RandomBrightnessContrast, but remove the unsupported HueSaturationValue.
        A.RandomBrightnessContrast(p=0.3)
    ])
    

def compute_enhanced_class_weights(
    dataset: BrainMRIDataset_Optimized, # Use the optimized dataset class
    num_classes=4,
    epsilon=1e-8,
    border_boost_factor=1.5
):
    """Calculates class weights based on frequency, presence, and borders.

    Args:
        dataset (BrainMRIDataset_Optimized): The training dataset instance.
        num_classes (int): The number of classes.
        epsilon (float): A small value to avoid division by zero.
        border_boost_factor (float): A factor to boost the weight of border pixels.

    Returns:
        torch.Tensor: A tensor of computed class weights.
    """
    
    # Initialize counters
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    border_pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    slice_presence_counts = torch.zeros(num_classes, dtype=torch.float64)

    # Access the dataset's file paths directly, avoiding the DataLoader
    patient_filepaths = dataset.patient_filepaths
    
    print("Starting enhanced weight computation...")
    for patient_id in tqdm(patient_filepaths.keys(), desc="Processing Patients for Weights"):
        # --- Load the 3D segmentation volume ONCE per patient ---
        seg_path = patient_filepaths[patient_id]['seg']
        seg_img = nib.load(seg_path)
        seg_data_3d = seg_img.get_fdata().astype(np.uint8)
        seg_data_3d[seg_data_3d == 4] = 3 # Remap label 4 to 3
        
        # --- 1. Aggregate total pixel counts from the 3D volume ---
        unique_labels, counts = np.unique(seg_data_3d, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label < num_classes:
                class_pixel_counts[label] += count
                
        # --- 2 & 3. Iterate through in-memory 2D slices for presence and borders ---
        # This is fast because the 3D volume is already in RAM.
        for z in range(seg_data_3d.shape[2]):
            mask_2d = seg_data_3d[:, :, z]
            
            for c in range(num_classes):
                class_mask = (mask_2d == c)
                if class_mask.any():
                    # Increment slice presence count if the class exists on this slice
                    slice_presence_counts[c] += 1
                    
                    # For tumor classes, calculate and count border pixels
                    if c > 0: # Classes 1, 2, 3
                        # Use 2D morphological operations as in the original logic
                        dilated = binary_dilation(class_mask)
                        eroded = binary_erosion(class_mask)
                        borders = dilated ^ eroded
                        border_pixel_counts[c] += borders.sum()

    print("--- Weight Calculation ---")
    print(f"Total Pixel Counts: {class_pixel_counts.numpy()}")
    print(f"Slice Presence Counts: {slice_presence_counts.numpy()}")
    print(f"Border Pixel Counts: {border_pixel_counts.numpy()}")

    # --- Combine the metrics using your original logic ---
    
    # Pixel frequency term (using sqrt for gentler weighting)
    freq_weights = 1.0 / torch.sqrt(class_pixel_counts + epsilon)
    
    # Sample presence term
    presence_weights = 1.0 / torch.sqrt(slice_presence_counts + epsilon)
    
    # Border boost term (for tumor classes 1, 2, 3)
    border_weights = torch.ones(num_classes, dtype=torch.float64)
    for c in [1, 2, 3]:
        if border_pixel_counts[c] > 0:
            # Boost weight for classes that have more non-border pixels than border pixels
            border_weights[c] = border_boost_factor * (class_pixel_counts[c] / border_pixel_counts[c])
            
    # Combine the different weighting strategies
    combined_weights = freq_weights * presence_weights * border_weights
    
    # Normalize the weights
    # Set background weight to be the median of the tumor weights to prevent it from vanishing
    if (combined_weights[1:] > 0).any():
         bg_weight = torch.median(combined_weights[1:][combined_weights[1:] > 0])
         combined_weights[0] = bg_weight
    
    # Final normalization so that weights sum to 1 (optional but good practice)
    final_weights = combined_weights / combined_weights.sum()
    
    print(f"Final Computed Weights: {final_weights.numpy()}")
    return final_weights.float() # Return as float32 for the model


def calculate_metrics(pred, target):
    """Calculates Dice and IoU scores for a batch.

    Args:
        pred (torch.Tensor): The model's prediction logits (B, C, H, W).
        target (torch.Tensor): The ground truth mask (B, H, W).

    Returns:
        tuple[float, float]: The mean Dice and IoU scores for the batch,
                             excluding the background class.
    """
    
    # Convert to class indices
    pred_mask = pred.argmax(dim=1)
    
    # Calculate Dice
    smooth = 1e-6
    pred_onehot = F.one_hot(pred_mask, num_classes=4).permute(0,3,1,2).float()
    target_onehot = F.one_hot(target, num_classes=4).permute(0,3,1,2).float()
    
    intersection = (pred_onehot * target_onehot).sum(dim=(2,3))
    union = pred_onehot.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Calculate IoU
    intersection = (pred_onehot * target_onehot).sum(dim=(2,3))
    union = pred_onehot.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    # Exclude background (class 0)
    return dice[:,1:].mean(), iou[:,1:].mean()


def compute_brats_hd95(model, loader, device, spacing=(1.0, 1.0), precision='fp16'): # Added spacing and precision params
    """Computes the 95th percentile Hausdorff Distance (HD95) for BraTS regions.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): The data loader for the validation set.
        device (str): The device to run inference on ('cuda' or 'cpu').
        spacing (tuple): Voxel spacing for distance calculation.
        precision (str): Model precision ('fp16' or 'fp32') for autocasting.

    Returns:
        dict: A dictionary with HD95 scores for 'et', 'tc', and 'wt' regions.
    """
    
    model.eval()
    hd95_lists = {'et': [], 'tc': [], 'wt': []}
    device_type = device.split(':')[0] # 'cuda' or 'cpu'

    with torch.no_grad():
        # Use tqdm for progress visibility during evaluation
        for x, y in tqdm(loader, desc="Calculating HD95", leave=False):
            x, y = x.to(device), y.to(device)

            # Use autocast consistent with training precision
            with torch.amp.autocast(device_type=device_type, enabled=(precision == 'fp16')):
                 pred_logits = model(x)

            pred = pred_logits.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()

            for p_idx, t_idx in zip(pred, true):
                # Define regions and corresponding labels for prediction (p) and truth (t)
                regions = {
                    'et': (p_idx == 3, t_idx == 3),                       # Enhancing Tumor (label 3)
                    'tc': ((p_idx == 1) | (p_idx == 3), (t_idx == 1) | (t_idx == 3)), # Tumor Core (labels 1 + 3)
                    'wt': (p_idx >= 1, t_idx >= 1)                        # Whole Tumor (labels 1 + 2 + 3)
                }

                for region, (pred_mask, true_mask) in regions.items():
                    has_pred = np.any(pred_mask)
                    has_true = np.any(true_mask)

                    if has_true and has_pred:
                        # Both prediction and ground truth have the tumor region
                        try:
                            score = medpy_metric.hd95(pred_mask, true_mask, voxelspacing=spacing)
                            hd95_lists[region].append(score)
                        except RuntimeError as e:
                            # Handle rare cases where medpy fails unexpectedly
                            print(f" Warning: medpy.hd95 failed for region {region} (sample): {e}. Appending inf.")
                            hd95_lists[region].append(np.inf)
                    elif has_true and not has_pred:
                        # False Negative: Ground truth exists, but prediction is empty
                        hd95_lists[region].append(np.inf)
                    elif not has_true and has_pred:
                        # False Positive: Prediction exists, but ground truth is empty
                        # BraTS evaluation often ignores HD95 here, but appending inf penalizes FPs strongly.
                        # Let's append inf for consistency in penalizing mismatches.
                        hd95_lists[region].append(np.inf)
                    # else: # not has_true and not has_pred
                        # True Negative: Both are empty. HD95 is not defined/calculated. Do nothing.


    # Calculate final mean HD95 scores, robustly handling infinity
    final_scores = {}
    for region, scores in hd95_lists.items():
        if not scores:
             # No valid samples or only empty cases found for this region
             final_scores[f'hd95_{region}'] = np.nan # Use NaN to indicate undefined mean
        else:
             scores_arr = np.array(scores, dtype=np.float64) # Use float64 for stability
             # Replace inf with nan for nanmean calculation
             scores_arr[np.isinf(scores_arr)] = np.nan
             mean_score = np.nanmean(scores_arr)
             # If mean_score is NaN (e.g., all scores were inf), report NaN
             final_scores[f'hd95_{region}'] = mean_score # Will be nan if no valid finite scores

    print(f"Computed HD95 raw means (NaN if undefined): ET={final_scores.get('hd95_et', 'N/A'):.2f}, TC={final_scores.get('hd95_tc', 'N/A'):.2f}, WT={final_scores.get('hd95_wt', 'N/A'):.2f}") # Add print for debug
    return final_scores


def evaluate(model, loader, criterion, device):
    """Performs a standard evaluation loop.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (str): The device to run evaluation on.

    Returns:
        tuple[float, float, float]: The average validation loss, Dice score, and IoU score.
    """
    
    model.eval()
    total_loss = 0.0
    dice_total = 0.0
    iou_total  = 0.0
    num_batches = len(loader)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            dice, iou = calculate_metrics(pred, y)
            dice_total += dice.item()
            iou_total  += iou.item()
    
    return total_loss / num_batches, \
           dice_total  / num_batches, \
           iou_total   / num_batches
           

def visualize_predictions(model, dataset, device, num_samples=10):
    """Displays model predictions against ground truth for visual inspection.

    Args:
        model (nn.Module): The trained model.
        dataset (Dataset): The dataset to draw samples from (e.g., validation set).
        device (str): The device for model inference.
        num_samples (int): The number of random samples to display.
    """
    
    # BraTS colormap: Background, Necrotic, Edema, Enhancing
    brats_cmap = plt.cm.colors.ListedColormap(['black', 'red', 'green', 'blue'])
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']

    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            x, y_true = dataset[idx]
            x_tensor = x.unsqueeze(0).to(device)
            y_pred = model(x_tensor).argmax(1).squeeze().cpu().numpy()
            y_true = y_true.numpy()

            # Get modalities
            flair = x[0].numpy()
            t1 = x[1].numpy()

            # Plot images
            axes[i,0].imshow(flair, cmap='gray')
            axes[i,0].set_title("FLAIR Input")
            axes[i,1].imshow(t1, cmap='gray')
            axes[i,1].set_title("T1 Input")
            axes[i,2].imshow(y_true, cmap=brats_cmap, vmin=0, vmax=3)
            axes[i,2].set_title("Ground Truth")
            axes[i,3].imshow(y_pred, cmap=brats_cmap, vmin=0, vmax=3)
            axes[i,3].set_title("Prediction")

            for ax in axes[i]: ax.axis('off')

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=brats_cmap(i), markersize=10,
                      label=class_names[i]) for i in range(4)]
    fig.legend(handles=legend_elements, loc='lower right', ncol=4)
    plt.tight_layout()
    plt.show()