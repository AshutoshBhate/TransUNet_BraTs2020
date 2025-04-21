# utils.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors
from medpy.metric import binary as medpy_metric  # For HD95 calculation

# --- Class Weight Computation ---

def compute_class_weights(dataset, epsilon=1e-8, border_boost=1.5, batch_size=8, num_workers=4):
    """Computes class weights considering pixel frequency, sample presence, and borders.

    This enhanced weight computation aims to balance classes in segmentation tasks,
    giving special attention to underrepresented classes and tumor borders. It uses
    square root balancing for less aggressive weighting compared to simple inverse
    frequency.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing (image, mask) pairs.
            It's assumed the dataset's __getitem__ returns a tuple where the
            second element is the mask tensor.
        epsilon (float, optional): A small value to prevent division by zero.
            Defaults to 1e-8.
        border_boost (float, optional): A factor to boost the weight of border pixels
            for tumor classes (1, 2, 3). Defaults to 1.5.
        batch_size (int, optional): Batch size for the DataLoader used internally.
            Defaults to 8.
        num_workers (int, optional): Number of workers for the DataLoader.
            Defaults to 4.


    Returns:
        torch.Tensor: A 1D tensor of size 4 containing the computed, normalized
            class weights. Index corresponds to class label (0: Background,
            1: Necrotic/Non-enhancing, 2: Edema, 3: Enhancing).
    """
    class_counts = torch.zeros(4, dtype=torch.float64) # Use float64 for accumulation
    border_counts = torch.zeros(4, dtype=torch.float64)
    sample_presence = torch.zeros(4, dtype=torch.float64)

    # Use a DataLoader to iterate efficiently, even if dataset is large
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Calculating class weights...")
    with torch.no_grad():
        for _, masks in tqdm(loader, desc="Computing weights"): # Assuming dataset returns (image, mask)
            # masks shape: (batch_size, H, W)

            # 1. Pixel-level counts
            for c in range(4):
                class_counts[c] += (masks == c).sum().item()

            # 2. Sample presence (count samples with > 100 pixels of a class)
            for c in range(4):
                # Check presence per sample in batch, then sum boolean results
                sample_presence[c] += ((masks == c).sum(dim=(1, 2)) > 100).sum().item()

            # 3. Border emphasis (only for tumor classes 1, 2, 3)
            # Process each mask in the batch individually for morphology ops
            for mask_idx in range(masks.shape[0]):
                mask_np = masks[mask_idx].numpy().astype(bool) # Convert single mask to numpy
                for c in [1, 2, 3]:  # Tumor classes
                    class_mask = (mask_np == c)
                    if np.any(class_mask): # Check if class c is present
                        # Morphological operations to find borders
                        dilated = binary_dilation(class_mask)
                        eroded = binary_erosion(class_mask)
                        borders = dilated ^ eroded # XOR finds pixels in dilation but not erosion
                        border_counts[c] += borders.sum() # Sum border pixels for this class

    # --- Weight Calculation ---

    # Base weights using inverse square root frequency
    freq_weights = 1.0 / torch.sqrt(class_counts + epsilon)

    # Adjust weights based on sample presence (helps rare classes present in few samples)
    # If a class appears in many samples, its weight is reduced; if rare, increased.
    # Using total samples N might be better: N / sqrt(sample_presence + epsilon)
    # However, using 1/sqrt provides relative weighting between classes based on presence.
    total_samples = len(dataset) # More robust presence weighting
    presence_weights = total_samples / torch.sqrt(sample_presence + epsilon)
    # Handle case where a class might have 0 presence despite pixel counts (e.g., only <100 pixels)
    presence_weights[sample_presence == 0] = 1.0 # Assign neutral weight if never significantly present


    # Apply border boost factor to tumor classes
    border_weights = torch.ones(4, dtype=torch.float64)
    for c in [1, 2, 3]:
        if border_counts[c] > 0:
            # Boost weight proportionally to how much non-border area there is vs border area
            border_weights[c] = border_boost * (class_counts[c] / (border_counts[c] + epsilon))
        # If no borders detected (e.g., very small regions), don't boost.

    # Combine the weighting strategies
    combined_weights = freq_weights * presence_weights * border_weights

    # Normalize weights: Set background (class 0) weight to the median of tumor weights
    # This prevents the background from dominating too much or too little.
    tumor_weights = combined_weights[1:]
    if len(tumor_weights[tumor_weights.isfinite()]) > 0: # Ensure there are finite weights to calculate median
         median_tumor_weight = torch.median(tumor_weights[tumor_weights.isfinite()])
         # Handle case where all tumor weights might be inf/nan if counts were zero
         if torch.isfinite(median_tumor_weight):
             combined_weights[0] = median_tumor_weight
         else:
             combined_weights[0] = 1.0 # Fallback if median is not finite
    else:
        combined_weights[0] = 1.0 # Fallback if no tumor classes exist or have counts

    # Final normalization: Make weights sum to 1 (or number of classes, depending on preference)
    final_weights = combined_weights / combined_weights.sum()

    print(f"Computed Class Weights: {final_weights.numpy()}")
    return final_weights.float() # Return as float32 for use in loss functions

# --- Basic Metrics Calculation (within training loop) ---

def calculate_metrics(pred, target, smooth=1e-6):
    """Calculates Dice and IoU scores for a batch, excluding the background class.

    Args:
        pred (torch.Tensor): The model output logits (before argmax).
            Shape: (N, C, H, W), where N is batch size, C is number of classes.
        target (torch.Tensor): The ground truth masks.
            Shape: (N, H, W), containing class indices.
        smooth (float, optional): Smoothing factor to prevent division by zero.
            Defaults to 1e-6.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - torch.Tensor: Mean Dice score over the batch for foreground classes (1, 2, 3).
            - torch.Tensor: Mean IoU score over the batch for foreground classes (1, 2, 3).
    """
    num_classes = pred.shape[1]
    pred_mask = pred.argmax(dim=1) # Get predicted class indices: (N, H, W)

    # Convert predictions and target to one-hot format for multi-class metrics
    # Shape: (N, H, W, C) -> (N, C, H, W)
    pred_onehot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Calculate intersection and union per class across spatial dimensions
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3)) # Shape: (N, C)
    pred_sum = pred_onehot.sum(dim=(2, 3)) # Shape: (N, C)
    target_sum = target_onehot.sum(dim=(2, 3)) # Shape: (N, C)

    # Dice Score = 2 * Intersection / (Sum of Pred + Sum of Target)
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth) # Shape: (N, C)

    # IoU Score = Intersection / (Union) = Intersection / (Pred + Target - Intersection)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth) # Shape: (N, C)

    # Calculate mean scores, excluding the background class (class 0)
    # Mean across classes [1, 2, 3], then mean across batch dimension N
    mean_dice = dice[:, 1:].mean()
    mean_iou = iou[:, 1:].mean()

    return mean_dice, mean_iou


# --- BraTS Specific Metrics (Hausdorff Distance) ---

def compute_brats_hd95(model, loader, device, spacing=(1.0, 1.0, 1.0), precision='fp16'):
    """Computes the 95th percentile Hausdorff Distance (HD95) for BraTS regions.

    Calculates HD95 separately for Enhancing Tumor (ET), Tumor Core (TC),
    and Whole Tumor (WT). Handles cases where ground truth or prediction
    for a region might be empty. Uses MedPy library for calculation.

    Note: Assumes input data represents 3D volumes or 2D slices. The `spacing`
          parameter is crucial for correct distance calculation. For 2D slices
          from a 3D volume, ensure spacing reflects the in-plane resolution.

    Args:
        model (torch.nn.Module): The trained segmentation model.
        loader (torch.utils.data.DataLoader): DataLoader for the validation/test set.
        device (torch.device or str): The device ('cuda' or 'cpu') to run inference on.
        spacing (tuple, optional): Voxel spacing for HD calculation.
            Order should correspond to the spatial dimensions of the input numpy arrays
            (e.g., (z, y, x) or (y, x) if using 2D). Defaults to (1.0, 1.0, 1.0).
        precision (str, optional): Training precision ('fp16' or 'fp32') to enable
            torch.amp.autocast during inference if needed. Defaults to 'fp16'.

    Returns:
        dict: A dictionary containing the average HD95 scores for each region:
            {'hd95_et': float, 'hd95_tc': float, 'hd95_wt': float}.
            Returns np.nan if no valid samples (where both pred and true masks
            are non-empty) are found for a specific region.
    """
    model.eval()
    hd95_lists = {'et': [], 'tc': [], 'wt': []} # Store HD95 values for each sample
    device_type = device.type if isinstance(device, torch.device) else device.split(':')[0] # 'cuda' or 'cpu'

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Calculating HD95", leave=False):
            x, y = x.to(device), y.to(device) # Input images and ground truth masks

            # Inference with automatic mixed precision context if enabled
            with torch.amp.autocast(device_type=device_type, enabled=(precision == 'fp16')):
                pred_logits = model(x)

            # Get predicted segmentation maps (class indices)
            pred_masks = pred_logits.argmax(dim=1).cpu().numpy() # (N, H, W) or (N, D, H, W)
            true_masks = y.cpu().numpy() # (N, H, W) or (N, D, H, W)

            # Process each sample in the batch
            for pred_mask_np, true_mask_np in zip(pred_masks, true_masks):
                # Define BraTS regions based on class labels:
                # ET: Enhancing Tumor (label 3 -> remapped from 4)
                # TC: Tumor Core (labels 1 + 3)
                # WT: Whole Tumor (labels 1 + 2 + 3)
                regions = {
                    'et': (pred_mask_np == 3, true_mask_np == 3),
                    'tc': ((pred_mask_np == 1) | (pred_mask_np == 3), (true_mask_np == 1) | (true_mask_np == 3)),
                    'wt': (pred_mask_np >= 1, true_mask_np >= 1) # Any non-background label
                }

                for region, (pred_region_mask, true_region_mask) in regions.items():
                    has_pred = np.any(pred_region_mask)
                    has_true = np.any(true_region_mask)

                    if has_true and has_pred:
                        # Both prediction and ground truth have the region: Calculate HD95
                        try:
                            # Ensure masks are boolean
                            pred_bool = pred_region_mask.astype(bool)
                            true_bool = true_region_mask.astype(bool)
                            score = medpy_metric.hd95(pred_bool, true_bool, voxelspacing=spacing)
                            hd95_lists[region].append(score)
                        except RuntimeError as e:
                            # MedPy can fail in rare cases (e.g., empty masks after processing)
                            print(f" Warning: medpy.hd95 failed for region {region} (sample). Error: {e}. Appending inf.")
                            hd95_lists[region].append(np.inf) # Penalize failure cases

                    elif has_true and not has_pred:
                        # False Negative: Ground truth exists, prediction missed it. Max penalty.
                        hd95_lists[region].append(np.inf)

                    elif not has_true and has_pred:
                        # False Positive: Prediction exists, ground truth doesn't. Max penalty.
                         # Standard BraTS often ignores HD for FP cases, but using inf provides a penalty.
                        hd95_lists[region].append(np.inf)

                    # else: # not has_true and not has_pred
                        # True Negative: Both masks are empty. HD95 is undefined/not calculated. Do nothing.


    # Calculate final mean HD95 scores, robustly handling infinity and empty lists
    final_scores = {}
    for region, scores in hd95_lists.items():
        key = f'hd95_{region}'
        if not scores:
             # No samples were processed for this region (e.g., only TN cases)
            final_scores[key] = np.nan # Indicate undefined mean
        else:
            scores_arr = np.array(scores, dtype=np.float64) # Use float64 for stability

            # Treat infinite values (FN/FP cases or calculation errors) appropriately.
            # Option 1: Exclude inf values using nanmean (standard practice)
            scores_arr[np.isinf(scores_arr)] = np.nan
            mean_score = np.nanmean(scores_arr) # Calculates mean ignoring NaNs

            # Option 2: Replace inf with a large penalty value before mean (if desired)
            # max_finite_score = np.nanmax(scores_arr[np.isfinite(scores_arr)]) if np.any(np.isfinite(scores_arr)) else 374 # Approx sqrt(128^2*3)
            # scores_arr[np.isinf(scores_arr)] = max_finite_score * 1.1 # Penalty larger than max observed finite
            # mean_score = np.nanmean(scores_arr) # Now includes penalties

            # If mean_score is still NaN (e.g., all scores were inf/nan), report NaN
            final_scores[key] = mean_score if not np.isnan(mean_score) else np.nan

    print(f" Computed HD95 means (NaN if undefined): "
          f"ET={final_scores.get('hd95_et', 'N/A'):.2f}, "
          f"TC={final_scores.get('hd95_tc', 'N/A'):.2f}, "
          f"WT={final_scores.get('hd95_wt', 'N/A'):.2f}")
    return final_scores


# --- Comprehensive Per-Sample BraTS Metrics ---

def compute_brats_metrics(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """Computes Dice, IoU, HD95, and HD for a single BraTS sample's regions.

    Calculates metrics for Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT)
    for a single prediction-target pair. Uses PyTorch for Dice/IoU and MedPy for
    Hausdorff distances.

    Args:
        pred_mask (torch.Tensor): The predicted segmentation mask (class indices).
            Shape: (H, W) or (D, H, W). Must be on the same device as target_mask.
        target_mask (torch.Tensor): The ground truth segmentation mask.
            Shape: (H, W) or (D, H, W). Must be on the same device as pred_mask.
        spacing (tuple, optional): Voxel spacing for HD calculation.
            Defaults to (1.0, 1.0, 1.0).

    Returns:
        dict: A dictionary containing metrics for the sample:
            {
                'dice_et': float, 'iou_et': float, 'hd95_et': float, 'hd_et': float,
                'dice_tc': float, 'iou_tc': float, 'hd95_tc': float, 'hd_tc': float,
                'dice_wt': float, 'iou_wt': float, 'hd95_wt': float, 'hd_wt': float,
            }
            HD values can be np.nan (if both masks empty or MedPy error) or
            np.inf (if one mask empty, the other isn't).
    """
    metrics = {}
    # Define labels for each BraTS region (assuming class 4 was remapped to 3)
    # ET: Enhancing Tumor (label 3)
    # TC: Tumor Core (labels 1:Necrotic/Non-enhancing + 3:Enhancing)
    # WT: Whole Tumor (labels 1:Necrotic/Non-enhancing + 2:Edema + 3:Enhancing)
    regions = {
        'et': ([3], [3]),                     # Pred labels, True labels
        'tc': ([1, 3], [1, 3]),
        'wt': ([1, 2, 3], [1, 2, 3])
    }
    device = pred_mask.device # Get device from input tensor

    for region, (pred_labels, true_labels) in regions.items():
        # Create binary masks for the current region using torch.isin for efficiency
        pred_bin = torch.isin(pred_mask, torch.tensor(pred_labels, device=device))
        true_bin = torch.isin(target_mask, torch.tensor(true_labels, device=device))

        # --- Calculate Dice and IoU using PyTorch ---
        intersection = torch.logical_and(pred_bin, true_bin).sum().float()
        pred_sum = pred_bin.sum().float()
        true_sum = true_bin.sum().float()
        union = pred_sum + true_sum - intersection # Union = A + B - A*B
        sum_ = pred_sum + true_sum

        # Handle edge cases for Dice/IoU
        if sum_ == 0:  # Both prediction and target are empty (True Negative)
            dice = torch.tensor(1.0, device=device) # Perfect agreement
            iou = torch.tensor(1.0, device=device)
        elif intersection == 0: # Either FN/FP (one empty) or no overlap
             dice = torch.tensor(0.0, device=device)
             iou = torch.tensor(0.0, device=device)
        else: # Standard case
            dice = (2 * intersection) / sum_
            iou = intersection / union

        # --- Calculate Hausdorff Distances using MedPy ---
        # Convert binary masks to numpy arrays for MedPy
        pred_np = pred_bin.cpu().numpy().astype(bool)
        true_np = true_bin.cpu().numpy().astype(bool)

        hd95 = np.nan
        hd = np.nan
        has_pred = np.any(pred_np)
        has_true = np.any(true_np)

        if has_pred or has_true: # At least one mask is not empty
            if not has_pred or not has_true:
                # False positive (pred=T, true=F) or false negative (pred=F, true=T)
                hd95 = np.inf # Assign infinite distance as penalty
                hd = np.inf
            else:
                # Both masks are non-empty, calculate HD and HD95
                try:
                    hd95 = medpy_metric.hd95(pred_np, true_np, voxelspacing=spacing)
                    hd = medpy_metric.hd(pred_np, true_np, voxelspacing=spacing)
                except RuntimeError:
                    # MedPy might fail (e.g., connectivity issues, empty masks after internal processing)
                    # Keep hd95 and hd as np.nan in this case
                    pass
        # else: # Both are empty (True Negative) -> hd95, hd remain np.nan

        # Store metrics for this region
        metrics.update({
            f'dice_{region}': dice.item(),
            f'iou_{region}': iou.item(),
            f'hd95_{region}': hd95,
            f'hd_{region}': hd
        })

    return metrics


# --- Full Evaluation Function ---

def evaluate_with_brats_metrics(model, loader, device, spacing=(1.0, 1.0, 1.0)):
    """Evaluates the model using comprehensive BraTS metrics (Dice, IoU, HD95, HD).

    Iterates through the data loader, computes per-sample BraTS metrics using
    `compute_brats_metrics`, and aggregates them. Averages Dice/IoU over all
    samples, and HD/HD95 only over samples where the metric was valid (finite).

    Args:
        model (torch.nn.Module): The trained segmentation model.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation set.
        device (torch.device or str): The device ('cuda' or 'cpu') for inference.
        spacing (tuple, optional): Voxel spacing for HD calculation.
            Defaults to (1.0, 1.0, 1.0).

    Returns:
        dict: A dictionary containing aggregated and averaged metrics:
            Includes 'dice_et/tc/wt', 'iou_et/tc/wt', 'hd95_et/tc/wt', 'hd_et/tc/wt',
            'avg_dice', 'avg_iou', 'avg_hd95', 'avg_hd'.
            HD averages are calculated excluding np.inf/np.nan values.
    """
    model.eval()
    # Initialize accumulators for sums and counts
    metrics_sum = {
        **{f'dice_{r}': 0.0 for r in ['et', 'tc', 'wt']},
        **{f'iou_{r}': 0.0 for r in ['et', 'tc', 'wt']},
        **{f'hd95_{r}': 0.0 for r in ['et', 'tc', 'wt']},
        **{f'hd_{r}': 0.0 for r in ['et', 'tc', 'wt']},
    }
    # Counts for averaging (total samples for Dice/IoU, valid samples for HD)
    counts = {'et': 0, 'tc': 0, 'wt': 0, 'total': 0}

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Full Evaluation", leave=False):
            x, y = x.to(device), y.to(device) # Input images and ground truth masks
            pred_logits = model(x)
            pred_masks = pred_logits.argmax(dim=1) # Shape: (N, H, W) or (N, D, H, W)

            # Process each sample in the batch
            for i in range(x.size(0)):
                # Calculate all metrics for the current sample
                sample_metrics = compute_brats_metrics(pred_masks[i], y[i], spacing)

                counts['total'] += 1 # Increment total sample count

                for region in ['et', 'tc', 'wt']:
                    # Accumulate Dice and IoU (always defined, 0 or 1 in edge cases)
                    metrics_sum[f'dice_{region}'] += sample_metrics[f'dice_{region}']
                    metrics_sum[f'iou_{region}'] += sample_metrics[f'iou_{region}']

                    # Accumulate HD metrics ONLY if they are finite and valid
                    hd95_val = sample_metrics[f'hd95_{region}']
                    hd_val = sample_metrics[f'hd_{region}']

                    # A finite value indicates a case where both pred & true were non-empty
                    # and the MedPy calculation succeeded.
                    if np.isfinite(hd95_val): # Checks for not inf AND not nan
                        metrics_sum[f'hd95_{region}'] += hd95_val
                        if np.isfinite(hd_val): # Usually HD is finite if HD95 is
                             metrics_sum[f'hd_{region}'] += hd_val
                        # Increment count only for valid HD calculation cases for this region
                        counts[region] += 1
                    # Cases with np.inf (FN/FP) or np.nan (TN/error) are not summed
                    # and do not increment the specific region count.

    # Calculate final average metrics
    final_metrics = {}
    for region in ['et', 'tc', 'wt']:
        total_count = counts['total']
        valid_hd_count = counts[region]

        # Average Dice and IoU over all samples
        final_metrics[f'dice_{region}'] = metrics_sum[f'dice_{region}'] / total_count if total_count > 0 else 0.0
        final_metrics[f'iou_{region}'] = metrics_sum[f'iou_{region}'] / total_count if total_count > 0 else 0.0

        # Average HD metrics only over samples where they were valid (finite)
        final_metrics[f'hd95_{region}'] = metrics_sum[f'hd95_{region}'] / valid_hd_count if valid_hd_count > 0 else np.nan
        final_metrics[f'hd_{region}'] = metrics_sum[f'hd_{region}'] / valid_hd_count if valid_hd_count > 0 else np.nan


    # Compute overall mean values across regions (using nanmean to ignore potential NaNs in HD)
    final_metrics['avg_dice'] = np.nanmean([final_metrics['dice_et'], final_metrics['dice_tc'], final_metrics['dice_wt']])
    final_metrics['avg_iou'] = np.nanmean([final_metrics['iou_et'], final_metrics['iou_tc'], final_metrics['iou_wt']])
    final_metrics['avg_hd95'] = np.nanmean([final_metrics['hd95_et'], final_metrics['hd95_tc'], final_metrics['hd95_wt']])
    final_metrics['avg_hd'] = np.nanmean([final_metrics['hd_et'], final_metrics['hd_tc'], final_metrics['hd_wt']])

    return final_metrics


# --- Visualization ---

def visualize_predictions(model, dataset, device, num_samples=5, img_channel_map=None):
    """Displays input modalities, ground truth, and model predictions for random samples.

    Selects random samples from the dataset and plots selected input modalities
    (e.g., FLAIR, T1) alongside the ground truth segmentation and the model's
    predicted segmentation. Uses a standard BraTS colormap.

    Args:
        model (torch.nn.Module): The trained segmentation model.
        dataset (torch.utils.data.Dataset): The dataset to sample from. Assumes
            __getitem__ returns (image_tensor, mask_tensor). The image tensor
            should have shape (C, H, W) or (C, D, H, W).
        device (torch.device or str): The device ('cuda' or 'cpu') for inference.
        num_samples (int, optional): The number of random samples to visualize.
            Defaults to 5.
        img_channel_map (dict, optional): A dictionary mapping display names to
            channel indices in the input image tensor. Example: {'FLAIR': 0, 'T1': 1}.
            If None, defaults to {'Input 0': 0, 'Input 1': 1}. Defaults to None.
    """
    if img_channel_map is None:
        img_channel_map = {'Input 0': 0, 'Input 1': 1} # Default channels to display

    # Define BraTS colormap and class names (assuming background=0, NCR/NET=1, ED=2, ET=3)
    brats_cmap = matplotlib.colors.ListedColormap(['black', 'red', 'green', 'blue'])
    class_names = ['Background', 'Necrotic/Non-Enh', 'Edema', 'Enhancing']
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5] # Boundaries for discrete colormap
    norm = matplotlib.colors.BoundaryNorm(bounds, brats_cmap.N)

    model.eval()
    num_modalities_to_show = len(img_channel_map)
    fig, axes = plt.subplots(num_samples, num_modalities_to_show + 2,
                             figsize=(5 * (num_modalities_to_show + 2), 5 * num_samples))

    # Ensure axes is always a 2D array for consistent indexing, even if num_samples=1
    if num_samples == 1:
        axes = axes.reshape(1, -1)


    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            x, y_true = dataset[idx] # x shape: (C, H, W) or (C, D, H, W), y_true shape: (H, W) or (D, H, W)

            # Handle potential 3D data (e.g., select middle slice)
            is_3d = x.ndim == 4
            slice_idx = x.shape[1] // 2 if is_3d else None # Middle slice index if 3D

            if is_3d:
                 print(f"Visualizing middle slice ({slice_idx}) of 3D sample {idx}")
                 x_slice = x[:, slice_idx, :, :] # Select middle slice for all channels
                 y_true_slice = y_true[slice_idx, :, :]
            else:
                 x_slice = x
                 y_true_slice = y_true

            # Perform inference on the selected slice/2D image
            x_tensor = x_slice.unsqueeze(0).to(device) # Add batch dimension -> (1, C, H, W)
            pred_logits = model(x_tensor)
            y_pred_slice = pred_logits.argmax(1).squeeze().cpu().numpy() # (H, W)

            y_true_np = y_true_slice.numpy() # (H, W)

            # Plot input modalities
            col_idx = 0
            for name, channel_idx in img_channel_map.items():
                img_modality = x_slice[channel_idx].cpu().numpy() # (H, W)
                axes[i, col_idx].imshow(img_modality, cmap='gray')
                axes[i, col_idx].set_title(f"Sample {idx}: {name}")
                axes[i, col_idx].axis('off')
                col_idx += 1

            # Plot Ground Truth
            axes[i, col_idx].imshow(y_true_np, cmap=brats_cmap, norm=norm)
            axes[i, col_idx].set_title(f"Sample {idx}: Ground Truth")
            axes[i, col_idx].axis('off')
            col_idx += 1

            # Plot Prediction
            axes[i, col_idx].imshow(y_pred_slice, cmap=brats_cmap, norm=norm)
            axes[i, col_idx].set_title(f"Sample {idx}: Prediction")
            axes[i, col_idx].axis('off')

    # Create a single legend for the figure
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=brats_cmap(i),
                                    label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(class_names), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap and make space for legend
    plt.show()