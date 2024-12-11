from tqdm import tqdm
from dataset import BrightfieldMicroscopyDataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2


def calculate_segmentation_metrics(predicted_mask, label_mask):
    """
    Calculate segmentation performance metrics.

    Args:
        predicted_mask (numpy.ndarray): Predicted binary segmentation mask
        label_mask (numpy.ndarray): Ground truth binary segmentation mask

    Returns:
        dict: Segmentation performance metrics
    """
    # Ensure masks are binary (0 or 1)
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    label_mask = (label_mask > 0).astype(np.uint8)

    # Pixel Accuracy: Proportion of correctly classified pixels
    pixel_accuracy = np.sum(predicted_mask == label_mask) / label_mask.size

    # Intersection over Union (IoU)
    intersection = np.logical_and(predicted_mask, label_mask)
    union = np.logical_or(predicted_mask, label_mask)
    iou = np.sum(intersection) / np.sum(union)

    # Dice Coefficient (F1 Score)
    dice_coefficient = (
        2 * np.sum(intersection) / (np.sum(predicted_mask) + np.sum(label_mask))
    )

    # Precision: Proportion of predicted positives that are actually positive
    true_positives = np.sum(np.logical_and(predicted_mask, label_mask))
    predicted_positives = np.sum(predicted_mask)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0

    # Recall: Proportion of actual positives that are correctly predicted
    actual_positives = np.sum(label_mask)
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    return {
        "pixel_accuracy": pixel_accuracy,
        "iou": iou,
        "dice_coefficient": dice_coefficient,
        "precision": precision,
        "recall": recall,
    }

def remove_repeating_pattern(input_image, threshold=0.1, iterations=1):
    """
    Removes repeating patterns from an image using FFT.

    Parameters:
    - image: 2D numpy array, the input image.
    - threshold: float, threshold for identifying peaks in the frequency domain.
    - iterations: int, number of iterations to refine the pattern removal.

    Returns:
    - cleaned_image: 2D numpy array, the image with the pattern removed.
    """
    image = input_image.copy()
    for _ in range(iterations):
        # Step 1: FFT to frequency domain
        f = fft2(image)
        fshift = fftshift(f)
        magnitude = np.abs(fshift)

        # Step 2: Identify pattern frequencies
        mask = magnitude > (threshold * np.max(magnitude))
        pattern_fshift = fshift * mask

        # Step 3: Reconstruct pattern using inverse FFT
        pattern = np.real(ifft2(ifftshift(pattern_fshift)))

        # Step 4: Subtract pattern from original image
        image = image - pattern

    return image

def preprocess_channel(
    channel,
    fft_threshold=0.01,
    fft_iterations=2,
    canny_thresh_min=70,
    canny_thresh_max=80,
    channel_dil=True,
    channel_dil_kernel=3,
    channel_dil_iter=1,
):
    """
    Preprocess a single channel image for segmentation.

    Args:
        channel (np.ndarray): Single channel image
        threshold (float, optional): Threshold for removing repeating patterns. Defaults to 0.01.
        iterations (int, optional): Number of iterations for pattern removal. Defaults to 2.

    Returns:
        np.ndarray: Preprocessed channel image
    """
    # Remove repeating pattern
    cleaned_channel = remove_repeating_pattern(
        channel, threshold=fft_threshold, iterations=fft_iterations
    )

    # Normalize to 0-255 range
    tensor_min = cleaned_channel.min()
    tensor_max = cleaned_channel.max()
    normalized_channel = (
        (cleaned_channel - tensor_min) / (tensor_max - tensor_min) * 255
    )
    normalized_channel = normalized_channel.astype(np.uint8)

    # Edge detection
    edged = cv2.Canny(normalized_channel, canny_thresh_min, canny_thresh_max)

    # Dilation
    if channel_dil:
        dilation_kernel = np.ones((channel_dil_kernel, channel_dil_kernel), np.uint8)
        edged = cv2.dilate(edged, dilation_kernel, iterations=channel_dil_iter)

    return edged


def post_process_segmentation(
    processed_image,
    use_otsu=True,
    otsu_blur_kernel=11,
    otsu_blur_iter=5,
    use_opening=True,
    opening_kernel=10,
    opening_iter=1,
):
    """
    Apply post-processing steps to the segmentation result.

    Args:
        processed_image (np.ndarray): Processed image to refine

    Returns:
        np.ndarray: Refined segmentation result
    """
    # Convert to absolute scale
    img = cv2.convertScaleAbs(processed_image)

    # Gaussian blur and Otsu's thresholding
    if use_otsu:
        blur = cv2.GaussianBlur(
            img, (otsu_blur_kernel, otsu_blur_kernel), otsu_blur_iter
        )
        _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening
    if use_opening:
        kernel = np.ones((opening_kernel, opening_kernel), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_iter)

    # Crop and remove border regions
    return crop_image_borders(img)


def crop_image_borders(image, crop_width=15, corner_width=100, corner_height=100):
    """
    Remove border and corner regions from the image.

    Args:
        image (np.ndarray): Input image
        crop_width (int, optional): Width of border to remove. Defaults to 15.
        corner_width (int, optional): Width of corner area to remove. Defaults to 100.
        corner_height (int, optional): Height of corner area to remove. Defaults to 100.

    Returns:
        np.ndarray: Image with borders and corners removed
    """
    cropped = image.copy()

    # Remove borders
    cropped[:crop_width, :] = 0
    cropped[-crop_width:, :] = 0
    cropped[:, :crop_width] = 0
    cropped[:, -crop_width:] = 0

    # Remove corner regions
    cropped[:corner_height, :corner_width] = 0
    cropped[:corner_height, -corner_width:] = 0
    cropped[-corner_height:, :corner_width] = 0
    cropped[-corner_height:, -corner_width:] = 0

    return cropped


def fft_segmentation(
    input_image,
    label,
    channels_to_calculate=None,
    
    fft_threshold=0.01,
    fft_iterations=2,
    canny_thresh_min=70,
    canny_thresh_max=80,
    channel_dil_kernel=3,
    channel_dil_iter=1,
    channel_dil=True,

    use_otsu=True,
    use_opening=True,
    otsu_blur_kernel=11,
    otsu_blur_iter=5,
    opening_kernel=10,
    opening_iter=1,
):
    """
    Perform FFT-based segmentation on an image.

    Args:
        input_image (torch.Tensor): Input image tensor
        label (torch.Tensor): Ground truth label
        channels_to_calculate (int, optional): Number of channels to process.
                                               Defaults to all channels.

    Returns:
        tuple: Processed image, original label, and segmentation metrics
    """
    # Convert to numpy and handle channel selection

    input_image_np = input_image.numpy()
    channels = input_image_np.shape[0]

    image_np = input_image_np.copy()

    if channels_to_calculate is None:
        channels_to_calculate = channels

    # Process channels and store them in a NumPy array
    processed_channels = np.zeros(
        (channels_to_calculate, image_np.shape[1], image_np.shape[2]), dtype=np.uint8
    )

    for i in range(channels_to_calculate):
        processed_channels[i] = preprocess_channel(
            channel=image_np[i],
            fft_threshold=fft_threshold,
            fft_iterations=fft_iterations,
            canny_thresh_min=canny_thresh_min,
            canny_thresh_max=canny_thresh_max,
            channel_dil=channel_dil,
            channel_dil_kernel=channel_dil_kernel,
            channel_dil_iter=channel_dil_iter,
        )

    # Average processed channels
    averaged_image = np.mean(processed_channels, axis=0)

    processed_image = post_process_segmentation(
        processed_image=averaged_image,
        use_otsu=use_otsu,
        otsu_blur_kernel=otsu_blur_kernel,
        otsu_blur_iter=otsu_blur_iter,
        use_opening=use_opening,
        opening_kernel=opening_kernel,
        opening_iter=opening_iter,
    )
    metrics = calculate_segmentation_metrics(processed_image, label.numpy())
    return processed_image, processed_channels, label, metrics
def perform_segmentation_evaluation(dataloader, print_interval=20, save_images=False, target_dir='data_processed', single_channel_target_dir='data_processed'):
    all_metrics = {
        "pixel_accuracy": [],
        "iou": [],
        "dice_coefficient": [],
        "precision": [],
        "recall": [],
    }

    incorrect_masks = 0
    incorrect_images = 0

    # Iterate through the dataloader
    for idx, (image, label, paths) in enumerate(
        tqdm(dataloader, desc="Segmentation Evaluation"), 1
    ):
        
        if not image.any():
            print(f"Missing images for index {idx}.")
            incorrect_images += 1
        elif not label.any():
            print(f"Missing labels for index {idx}.")
            incorrect_masks += 1
            

        image = image[0]
        label = label[0]

        # Perform segmentation based on the chosen method
        postprocessed_image, preprocessed_channels, label, metrics = fft_segmentation(input_image=image,
                                                                                    label=label,
                                                                                    channels_to_calculate=None,

                                                                                    #preprocessing
                                                                                    fft_threshold=0.01,
                                                                                    fft_iterations=2,
                                                                                    canny_thresh_min=70,
                                                                                    canny_thresh_max=80,
                                                                                    channel_dil_kernel=3,
                                                                                    channel_dil_iter=1,
                                                                                    channel_dil=True,

                                                                                    #postprocessing
                                                                                    use_otsu=True,
                                                                                    otsu_blur_kernel=11,
                                                                                    otsu_blur_iter=5,
                                                                                    use_opening=True,
                                                                                    opening_kernel=10,
                                                                                    opening_iter=1,)

        if save_images:
            for i in range(preprocessed_channels.shape[0]):
                # Create full directory path
                full_save_dir = os.path.dirname(target_dir + paths[i][0][4:])
                os.makedirs(full_save_dir, exist_ok=True)
                
                # Save preprocessed channels
                image = Image.fromarray(preprocessed_channels[i])
                path = target_dir + paths[i][0][4:]
                image.save(path)
            
            # Save postprocessed image
            full_save_dir = os.path.dirname(single_channel_target_dir + paths[i][0].split('z')[0][4:])
            os.makedirs(full_save_dir, exist_ok=True)
            
            image = Image.fromarray(postprocessed_image)
            path = single_channel_target_dir + paths[i][0].split('z')[0][4:]
            image.save(f"{path}.tif")

        # Calculate metrics for this batch
        # batch_metrics = calculate_segmentation_metrics(predicted_mask, label.numpy())

        # Store metrics
        for metric_name, metric_value in metrics.items():
            all_metrics[metric_name].append(metric_value)

        # Print intermediate metrics at specified interval
        if idx % print_interval == 0:
            print(f"\nIntermediate Metrics at Image {idx}:")
            current_aggregated_metrics = {
                metric_name: np.mean(metric_values)
                for metric_name, metric_values in all_metrics.items()
            }

            for metric_name, metric_value in current_aggregated_metrics.items():
                print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.4f}")

    # Calculate final average metrics
    aggregated_metrics = {
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in all_metrics.items()
    }

    # Calculate metric variance for additional insight
    metric_variance = {
        metric_name: np.var(metric_values)
        for metric_name, metric_values in all_metrics.items()
    }

    # Print final detailed results
    print("\nFinal Segmentation Metrics Summary:")
    for metric_name, metric_value in aggregated_metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.4f}")

    print("\nFinal Metric Variance:")
    for metric_name, variance in metric_variance.items():
        print(f"{metric_name.replace('_', ' ').title()} Variance: {variance:.6f}")

    print(f"incorrect images: {incorrect_images}")
    print(f"incorrect masks: {incorrect_masks}")

    return {
        "mean_metrics": aggregated_metrics,
        "metric_variance": metric_variance,
        "per_batch_metrics": all_metrics,
    }