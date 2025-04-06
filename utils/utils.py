"""
Utility functions for computing class weights and evaluation metrics
for brain tumor segmentation (specifically tailored for BraTS).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# --- Class Weight Computation ---

def compute_class_weights(dataset: Dataset, num_classes: int = 4, epsilon: float = 1e-8,
                          border_boost: float = 1.5, batch_size: int = 16, num_workers: int = 0):
    """Compute combined class weights with border emphasis for segmentation tasks.

    Accounts for pixel frequency, sample presence, and border emphasis.

    Args:
        dataset (Dataset): Dataset instance returning (image, mask) or just masks.
        num_classes (int): Total number of segmentation classes. Defaults to 4.
        epsilon (float): Small value to avoid division by zero. Defaults to 1e-8.
        border_boost (float): Factor to boost weights for border pixels. Defaults to 1.5.
        batch_size (int): Batch size for iterating through the dataset. Defaults to 16.
        num_workers (int): Number of workers for the temporary DataLoader. Defaults to 0.

    Returns:
        torch.Tensor: Normalized class weights, shape [num_classes].
    """
    print(f"Computing class weights for {num_classes} classes...")
    if num_classes != 4:
        print("Warning: Class weight computation logic is optimized for BraTS 4 classes (BG, NCR, ED, ET).")

    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    border_counts = torch.zeros(num_classes, dtype=torch.float64)
    sample_presence = torch.zeros(num_classes, dtype=torch.float64)

    # Use a DataLoader with specified parameters
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    struct_el = None # Default structure element for morphology

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing masks for weights"):
            # Handle cases where dataset yields tuples (img, mask) or just masks
            if isinstance(batch, (list, tuple)):
                masks = batch[1]
            else:
                masks = batch

            masks_np = masks.cpu().numpy()

            for i in range(masks_np.shape[0]):
                mask_np = masks_np[i]

                # Pixel counts and sample presence
                for c in range(num_classes):
                    class_pixels = (mask_np == c)
                    pixel_count = class_pixels.sum()
                    class_counts[c] += pixel_count
                    if pixel_count > 100:
                        sample_presence[c] += 1

                # Border emphasis (assuming classes 1, 2, 3 are foreground/tumor)
                # Adjust this range if num_classes != 4
                foreground_classes = range(1, num_classes)
                for c in foreground_classes:
                    class_mask = (mask_np == c)
                    if class_mask.any():
                        try:
                            dilated = binary_dilation(class_mask, structure=struct_el)
                            eroded = binary_erosion(class_mask, structure=struct_el)
                            borders = np.logical_xor(dilated, eroded)
                            border_counts[c] += borders.sum()
                        except Exception as e:
                             print(f"Warning: Morphological operation failed for class {c}. Error: {e}")

    print("\n--- Raw Counts ---")
    print(f"Pixels/class: {class_counts.numpy()}")
    print(f"Slices present (>100px): {sample_presence.numpy()}")
    print(f"Border Pixels (FG only): {border_counts.numpy()}")

    # Calculate component weights
    freq_weights = 1.0 / torch.sqrt(class_counts + epsilon)
    presence_weights = 1.0 / torch.sqrt(sample_presence + epsilon)
    border_weights = torch.ones(num_classes, dtype=torch.float64)
    foreground_classes = range(1, num_classes)
    for c in foreground_classes:
        if border_counts[c] > 0 and class_counts[c] > 0:
             border_ratio_boost = (class_counts[c] / (border_counts[c] + epsilon))
             # Log scaling helps moderate the boost
             border_weights[c] = 1.0 + border_boost * torch.log1p(border_ratio_boost)

    combined_weights = freq_weights * presence_weights * border_weights

    # Normalize background weight (class 0)
    foreground_weights = combined_weights[1:]
    if len(foreground_weights) > 0 and torch.sum(foreground_weights > 0) > 0 :
        valid_fg_weights = foreground_weights[foreground_weights > 0]
        if len(valid_fg_weights) > 0:
             median_fg_weight = torch.median(valid_fg_weights)
             combined_weights[0] = median_fg_weight
        else:
             combined_weights[0] = 1.0
    else:
         combined_weights[0] = 1.0

    # Normalize final weights (e.g., scale to sum to num_classes)
    normalized_weights = (combined_weights / combined_weights.sum()) * num_classes

    print("\n--- Computed Weights ---")
    print(f"Final Normalized Weights: {normalized_weights.numpy()}")
    print("-" * 20)

    return normalized_weights.float()


# --- Basic Foreground Metrics ---

def calculate_metrics(pred_logits, target, num_classes: int = 4, smooth: float = 1e-6):
    """Calculate average Dice and IoU over foreground classes.

    Args:
        pred_logits (torch.Tensor): Model output logits [B, C, H, W].
        target (torch.Tensor): Ground truth masks [B, H, W].
        num_classes (int): Total number of classes. Defaults to 4.
        smooth (float): Smoothing factor. Defaults to 1e-6.

    Returns:
        tuple: Average foreground Dice score (float), Average foreground IoU score (float).
    """
    pred_mask = torch.argmax(pred_logits, dim=1)
    pred_onehot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    pred_sum = pred_onehot.sum(dim=(2, 3))
    target_sum = target_onehot.sum(dim=(2, 3))

    # Dice
    dice_numerator = 2. * intersection + smooth
    dice_denominator = pred_sum + target_sum + smooth
    dice_scores = dice_numerator / dice_denominator

    # IoU
    iou_numerator = intersection + smooth
    iou_denominator = pred_sum + target_sum - intersection + smooth
    iou_scores = iou_numerator / iou_denominator

    # Average over foreground classes (classes 1 to num_classes-1) and batch
    mean_dice = dice_scores[:, 1:].mean()
    mean_iou = iou_scores[:, 1:].mean()

    # Handle potential NaN if no foreground pixels exist in batch
    if torch.isnan(mean_dice): mean_dice = torch.tensor(0.0, device=pred_logits.device)
    if torch.isnan(mean_iou): mean_iou = torch.tensor(0.0, device=pred_logits.device)

    return mean_dice.item(), mean_iou.item()


# --- BraTS Specific Metrics ---

def compute_brats_metrics(pred_mask, target_mask, num_classes: int = 4, eps: float = 1e-8):
    """Computes Dice and IoU for BraTS regions (ET, TC, WT) for a single sample.

    Args:
        pred_mask (Tensor): Predicted mask tensor [H, W] with class indices (0-3).
        target_mask (Tensor): Ground truth mask tensor [H, W] with class indices (0-3).
        num_classes (int): Number of classes (must be 4 for BraTS logic). Defaults to 4.
        eps (float): Epsilon for numerical stability. Defaults to 1e-8.

    Returns:
        tuple: ( (dice_et, dice_tc, dice_wt), (iou_et, iou_tc, iou_wt) ) as tensors.
    """
    if num_classes != 4:
        raise ValueError("compute_brats_metrics requires num_classes=4 for BraTS regions")

    # Create boolean masks for each class from integer masks
    # Class indices: 0=BG, 1=NCR/NET, 2=ED, 3=ET
    pred_c1 = (pred_mask == 1)
    target_c1 = (target_mask == 1)
    pred_c2 = (pred_mask == 2)
    target_c2 = (target_mask == 2)
    pred_c3 = (pred_mask == 3)
    target_c3 = (target_mask == 3)

    # Define BraTS regions
    pred_et = pred_c3
    target_et = target_c3
    pred_tc = pred_c1 | pred_c3
    target_tc = target_c1 | target_c3
    pred_wt = pred_c1 | pred_c2 | pred_c3
    target_wt = target_c1 | target_c2 | target_c3

    # --- Helper function to calculate Dice and IoU for a region ---
    def _calculate_dice_iou(pred_region, target_region):
        intersection = torch.logical_and(pred_region, target_region).float().sum()
        pred_sum = pred_region.float().sum()
        target_sum = target_region.float().sum()
        union = torch.logical_or(pred_region, target_region).float().sum()

        dice_num = 2. * intersection + eps
        dice_den = pred_sum + target_sum + eps
        dice = dice_num / dice_den

        iou_num = intersection + eps
        iou_den = union + eps
        iou = iou_num / iou_den

        # Handle case where target region is empty
        if target_sum == 0:
            # Score is 1 if prediction is also empty, 0 otherwise
            score = torch.tensor(1.0 if pred_sum == 0 else 0.0, device=pred_region.device)
            dice = score
            iou = score

        return dice, iou

    # Calculate metrics for each region
    dice_et, iou_et = _calculate_dice_iou(pred_et, target_et)
    dice_tc, iou_tc = _calculate_dice_iou(pred_tc, target_tc)
    dice_wt, iou_wt = _calculate_dice_iou(pred_wt, target_wt)

    return (dice_et, dice_tc, dice_wt), (iou_et, iou_tc, iou_wt)

# --- BraTS Evaluation Function ---

def evaluate_brats_regions(model, loader, device, num_classes: int = 4, precision: str = 'fp32'):
    """Evaluates the model using BraTS specific metrics (ET, TC, WT Dice/IoU).

    Args:
        model (nn.Module): The trained segmentation model.
        loader (DataLoader): DataLoader for the validation or test set.
        device (torch.device): The device to run evaluation on.
        num_classes (int): Number of classes (must be 4). Defaults to 4.
        precision (str): 'fp16' or 'fp32' for autocast. Defaults to 'fp32'.

    Returns:
        dict: Dictionary containing average metrics over the dataset.
    """
    if num_classes != 4:
        raise ValueError("evaluate_brats_regions requires num_classes=4")

    model.eval()
    metrics = {
        'dice_et': 0.0, 'dice_tc': 0.0, 'dice_wt': 0.0,
        'iou_et': 0.0, 'iou_tc': 0.0, 'iou_wt': 0.0
    }
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating BraTS Regions", leave=False):
            images, masks_true = batch
            images = images.to(device, non_blocking=True)
            masks_true = masks_true.to(device, non_blocking=True)

            # Use autocast for consistency during inference
            with torch.cuda.amp.autocast(enabled=(precision == 'fp16')):
                pred_logits = model(images)

            masks_pred = torch.argmax(pred_logits, dim=1) # Get predicted class indices [B, H, W]

            for i in range(images.size(0)):
                dice_scores, iou_scores = compute_brats_metrics(
                    masks_pred[i], masks_true[i], num_classes=num_classes
                )
                metrics['dice_et'] += dice_scores[0].item()
                metrics['dice_tc'] += dice_scores[1].item()
                metrics['dice_wt'] += dice_scores[2].item()
                metrics['iou_et'] += iou_scores[0].item()
                metrics['iou_tc'] += iou_scores[1].item()
                metrics['iou_wt'] += iou_scores[2].item()
                total_samples += 1

    avg_metrics = {k: v / total_samples for k, v in metrics.items() if total_samples > 0}

    if total_samples > 0:
        avg_metrics['avg_dice'] = (avg_metrics['dice_et'] + avg_metrics['dice_tc'] + avg_metrics['dice_wt']) / 3
        avg_metrics['avg_iou'] = (avg_metrics['iou_et'] + avg_metrics['iou_tc'] + avg_metrics['iou_wt']) / 3
    else:
        avg_metrics['avg_dice'] = 0.0
        avg_metrics['avg_iou'] = 0.0
        print("Warning: No samples processed during BraTS evaluation.")

    return avg_metrics

# --- Visualization ---

def visualize_predictions(model, dataset, device, num_samples: int = 5, num_classes: int = 4):
    """Displays input modalities, Ground Truth, and Prediction for random samples.

    Args:
        model (nn.Module): The trained model.
        dataset (Dataset): The dataset to sample from (e.g., validation set).
        device (torch.device): Device for model inference.
        num_samples (int): Number of random samples to visualize. Defaults to 5.
        num_classes (int): Number of segmentation classes. Defaults to 4.
    """
    if num_classes != 4:
        print("Warning: Visualization colormap and titles assume 4 BraTS classes.")

    # BraTS colormap: 0:Black, 1:Red(NCR/NET), 2:Green(ED), 3:Blue(ET)
    brats_colors = ['black', 'red', 'green', 'blue']
    cmap = colors.ListedColormap(brats_colors[:num_classes])
    class_names = ['Background', 'NCR/NET', 'Edema', 'Enhancing Tumor']

    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples), squeeze=False)
    fig.suptitle("Model Predictions vs Ground Truth", fontsize=16)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                image_tensor, mask_true_tensor = dataset[idx]
                input_tensor = image_tensor.unsqueeze(0).to(device)

                pred_logits = model(input_tensor)
                pred_mask_tensor = torch.argmax(pred_logits, dim=1).squeeze(0).cpu()

                mask_pred_np = pred_mask_tensor.numpy()
                mask_true_np = mask_true_tensor.numpy()
                # Assuming channel order: 0:FLAIR, 1:T1, 2:T1ce, 3:T2
                flair_np = image_tensor[0].numpy()
                t1ce_np = image_tensor[2].numpy() # T1ce often shows contrast enhancement

                ax_row = axes[i]
                ax_row[0].imshow(flair_np, cmap='gray')
                ax_row[0].set_title(f"Sample {idx}: FLAIR Input")
                ax_row[1].imshow(t1ce_np, cmap='gray')
                ax_row[1].set_title(f"Sample {idx}: T1ce Input")
                ax_row[2].imshow(mask_true_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
                ax_row[2].set_title(f"Sample {idx}: Ground Truth")
                ax_row[3].imshow(mask_pred_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
                ax_row[3].set_title(f"Sample {idx}: Prediction")

                for ax in ax_row: ax.axis('off')

            except Exception as e:
                print(f"Error visualizing sample index {idx}: {e}")
                # Optionally turn off axes for the failed row
                for ax in axes[i]: ax.axis('off')
                axes[i,0].set_title(f"Error loading/processing sample {idx}")


    # Create figure legend
    legend_patches = [plt.Rectangle((0,0),1,1,fc=cmap.colors[j]) for j in range(num_classes)]
    fig.legend(legend_patches, class_names[:num_classes], loc='lower center', ncol=num_classes, fontsize='large')

    plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Adjust layout
    # Consider saving the figure instead of showing interactively in non-interactive environments
    # plt.savefig("predictions_visualization.png")
    # print("Prediction visualization saved to predictions_visualization.png")
    plt.show()