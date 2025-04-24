import torch
import numpy as np
import kornia
import logging
from collections import deque
from PySide6.QtGui import QImage

# Check if GPU is available
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IS_GPU_AVAILABLE = DEVICE.type == "cuda"
except Exception as e:
    DEVICE = torch.device("cpu")
    IS_GPU_AVAILABLE = False
    logging.error(f"Error initializing GPU: {e}")

# Configure logging
logger = logging.getLogger(__name__)

class GpuBufferPool:
    """
    A pool of pre-allocated GPU buffers to minimize memory allocation overhead.
    Buffers are allocated based on image dimensions and reused when possible.
    """
    def __init__(self, max_buffers=10):
        self.max_buffers = max_buffers
        # Dictionary of buffer pools keyed by (height, width, channels)
        self.buffers = {}
        logger.info(f"GPU Buffer Pool initialized with max_buffers={max_buffers}, device={DEVICE}")
        
    def get_buffer(self, height, width, channels=3, dtype=torch.float32):
        """
        Get a GPU buffer for the specified dimensions.
        
        Args:
            height (int): Height of the buffer
            width (int): Width of the buffer
            channels (int): Number of channels
            dtype (torch.dtype): Data type of the buffer
            
        Returns:
            torch.Tensor: A tensor of the specified dimensions on the GPU
        """
        if not IS_GPU_AVAILABLE:
            # Return CPU tensor if GPU is not available
            return torch.empty((height, width, channels), dtype=dtype)
            
        key = (height, width, channels, dtype)
        
        # Create pool for this key if it doesn't exist
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=self.max_buffers)
            
        # Get a buffer from the pool or create a new one
        if not self.buffers[key]:
            logger.debug(f"Creating new GPU buffer {key}")
            return torch.empty((height, width, channels), dtype=dtype, device=DEVICE)
        else:
            logger.debug(f"Reusing GPU buffer {key}")
            return self.buffers[key].pop()
            
    def return_buffer(self, buffer):
        """
        Return a buffer to the pool for reuse.
        
        Args:
            buffer (torch.Tensor): Buffer to return to the pool
        """
        if not IS_GPU_AVAILABLE or buffer.device.type != "cuda":
            return  # Don't store CPU tensors
            
        # Determine the key from the buffer's properties
        key = (buffer.shape[0], buffer.shape[1], buffer.shape[2], buffer.dtype)
        
        # Add the buffer back to the pool if we have a pool for this key
        if key in self.buffers and len(self.buffers[key]) < self.max_buffers:
            self.buffers[key].append(buffer)

# Create a global buffer pool instance
buffer_pool = GpuBufferPool()

def upload_tensor(img_np, non_blocking=True, to_bchw=False):
    """
    Upload a numpy image to a GPU tensor.
    
    Args:
        img_np (numpy.ndarray): Numpy array containing the image
        non_blocking (bool): Whether to perform the upload asynchronously
        to_bchw (bool): Whether to automatically convert tensor to [B, C, H, W] format
        
    Returns:
        torch.Tensor: Tensor on the GPU
    """
    if img_np is None:
        return None
        
    if not IS_GPU_AVAILABLE:
        tensor = torch.from_numpy(img_np)
        # If requested, convert to BCHW format before returning
        if to_bchw and len(img_np.shape) == 3:  # [H, W, C]
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        return tensor
        
    try:
        # Get height, width, channels from input image
        if len(img_np.shape) == 3:
            h, w, c = img_np.shape
        else:
            h, w = img_np.shape
            c = 1
            
        # Use float32 for most processing
        tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
        
        # Reshape if needed
        if c == 1:
            tensor = tensor.unsqueeze(-1)
        
        # Convert to [B, C, H, W] format if requested
        if to_bchw and len(tensor.shape) == 3:  # [H, W, C]
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            
        # Upload to GPU
        return tensor.to(DEVICE, non_blocking=non_blocking)
    except Exception as e:
        logger.error(f"Error uploading tensor to GPU: {e}")
        tensor = torch.from_numpy(img_np)
        # If requested, convert to BCHW format before returning
        if to_bchw and len(img_np.shape) == 3:  # [H, W, C]
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        return tensor

def upload_tensor_async(img_np):
    """
    Upload a numpy image to a GPU tensor asynchronously.
    
    Args:
        img_np (numpy.ndarray): Numpy array containing the image
        
    Returns:
        torch.Tensor: Tensor on the GPU
    """
    return upload_tensor(img_np, non_blocking=True)

def download_tensor(tensor):
    """
    Download a GPU tensor to a numpy array.
    
    Args:
        tensor (torch.Tensor): Tensor on the GPU
        
    Returns:
        numpy.ndarray: Numpy array containing the image
    """
    if tensor is None:
        return None
        
    try:
        # Move to CPU and convert to numpy
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
            
        # Scale to 0-255 range and convert to uint8
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
            
        np_array = tensor.numpy().astype(np.uint8)
        
        # Squeeze if single channel
        if np_array.shape[-1] == 1:
            np_array = np.squeeze(np_array, axis=-1)
            
        return np_array
    except Exception as e:
        logger.error(f"Error downloading tensor from GPU: {e}")
        return tensor.cpu().numpy()

def download_tensor_async(tensor):
    """
    Download a GPU tensor to a numpy array asynchronously.
    
    Args:
        tensor (torch.Tensor): Tensor on the GPU
        
    Returns:
        numpy.ndarray: Numpy array containing the image
    """
    if tensor is None:
        return None
        
    # Create a cuda stream
    if IS_GPU_AVAILABLE and tensor.device.type == "cuda":
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = download_tensor(tensor)
        # Synchronize to ensure download is complete
        stream.synchronize()
        return result
    else:
        return download_tensor(tensor)

def create_hsv_mask_gpu(image_tensor, hsv_values, device=None):
    """
    Create an HSV mask for an image tensor using GPU acceleration.
    Optimized for performance with minimized memory transfers and operations.
    Efficiently handles both single images and batches.
    
    Args:
        image_tensor (torch.Tensor): Input tensor in RGB format 
                                    - Supports [B, C, H, W], [C, H, W], or [H, W, C] formats
        hsv_values (dict): Dictionary containing HSV min/max values:
                          - h_min, h_max: Hue range (0-179)
                          - s_min, s_max: Saturation range (0-255)
                          - v_min, v_max: Value range (0-255)
        device (torch.device): Device to use (defaults to global DEVICE)
        
    Returns:
        torch.Tensor: Binary mask tensor or None on error
    """
    if image_tensor is None:
        logger.warning("Invalid tensor for HSV mask creation")
        return None
        
    try:
        # Use provided device or default to global DEVICE
        device = device or DEVICE
        
        # Ensure tensor is on the correct device
        if image_tensor.device != device:
            image_tensor = image_tensor.to(device)
        
        # Process tensor shape to ensure it's in the correct format [B, C, H, W]
        shape_len = len(image_tensor.shape)
        
        # Save original input format for later
        is_batched = shape_len == 4
        
        # Handle different input formats
        if shape_len == 4:  # [B, C, H, W] or [B, H, W, C]
            # Check if format is [B, H, W, C] and convert if needed
            if image_tensor.shape[1] > 4 and image_tensor.shape[3] <= 4:
                logger.debug(f"Converting tensor from [B, H, W, C] to [B, C, H, W]: {image_tensor.shape}")
                image_tensor = image_tensor.permute(0, 3, 1, 2)
            # Verify that channels are now in correct position
            if image_tensor.shape[1] > 4:
                raise ValueError(f"Tensor shape looks incorrect: {image_tensor.shape}. Expected channels dimension to be ≤ 4.")
        elif shape_len == 3:  # [C, H, W] or [H, W, C]
            # Check if format is [H, W, C] and convert if needed
            if image_tensor.shape[0] > 4 and image_tensor.shape[2] <= 4:
                logger.debug(f"Converting tensor from [H, W, C] to [C, H, W]: {image_tensor.shape}")
                image_tensor = image_tensor.permute(2, 0, 1)
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        else:
            raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")
        
        logger.debug(f"Processing tensor with shape: {image_tensor.shape}")
        
        # Get batch size for memory optimization
        batch_size = image_tensor.shape[0]
        
        # Process in smaller batches if needed (for very large batches)
        max_batch_size = 8
        
        if batch_size > max_batch_size:
            # Process in smaller batches
            logger.debug(f"Processing large batch in chunks: {batch_size} images")
            results = []
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                batch_result = create_hsv_mask_gpu(
                    image_tensor[i:end_idx], hsv_values, device
                )
                if batch_result is not None:
                    results.append(batch_result)
            
            if results:
                # Concatenate results along batch dimension
                return torch.cat(results, dim=0)
            else:
                return None
        
        # Convert RGB to HSV - this is the most expensive operation
        with torch.no_grad():  # Use no_grad for inference to reduce memory usage
            hsv_tensor = kornia.color.rgb_to_hsv(image_tensor)
            logger.debug(f"HSV tensor shape: {hsv_tensor.shape}")
        
        # Extract HSV values and normalize
        h_min = hsv_values["h_min"] / 180.0
        h_max = hsv_values["h_max"] / 180.0
        s_min = hsv_values["s_min"] / 255.0
        s_max = hsv_values["s_max"] / 255.0
        v_min = hsv_values["v_min"] / 255.0
        v_max = hsv_values["v_max"] / 255.0
        
        # Create mask using vector operations for better efficiency
        # Extract channels for better memory locality
        h_channel = hsv_tensor[:, 0:1, :, :]
        s_channel = hsv_tensor[:, 1:2, :, :]
        v_channel = hsv_tensor[:, 2:3, :, :]
        
        # Create masks with optimized operations
        with torch.no_grad():  # Use no_grad for inference
            # Handle hue wrap-around case more efficiently
            if h_min <= h_max:
                # Standard case: single range
                h_mask = torch.logical_and(h_channel >= h_min, h_channel <= h_max)
            else:
                # Wrap-around case: two ranges combined
                h_mask = torch.logical_or(h_channel >= h_min, h_channel <= h_max)
            
            # Apply saturation and value constraints
            s_mask = torch.logical_and(s_channel >= s_min, s_channel <= s_max)
            v_mask = torch.logical_and(v_channel >= v_min, v_channel <= v_max)
            
            # Combine masks in a single operation for efficiency
            combined_mask = torch.logical_and(torch.logical_and(h_mask, s_mask), v_mask)
            
            # Convert to binary mask (0 or 255)
            mask = combined_mask.to(torch.uint8) * 255
            
            # Free up memory
            del h_mask, s_mask, v_mask, combined_mask
            
            # Apply morphological operations for mask cleaning
            if torch.any(mask):
                # Create a 2D kernel for morphological operations
                kernel_size = 5
                kernel = torch.ones(kernel_size, kernel_size, device=device)
                
                # Process each batch item separately because kornia morphology
                # operations expect 2D kernels for 2D data
                masks_list = []
                for i in range(mask.shape[0]):
                    # Process each channel separately
                    channels_list = []
                    for c in range(mask.shape[1]):
                        # Extract 2D mask
                        mask_2d = mask[i, c].float()
                        
                        # Apply erosion
                        eroded = kornia.morphology.erosion(mask_2d.unsqueeze(0).unsqueeze(0), kernel)
                        
                        # Apply dilation (twice)
                        dilated = kornia.morphology.dilation(eroded, kernel)
                        dilated = kornia.morphology.dilation(dilated, kernel)
                        
                        # Squeeze back to 2D and add to channel list
                        channels_list.append(dilated.squeeze(0).squeeze(0))
                    
                    # Stack channels back together
                    if len(channels_list) > 1:
                        mask_processed = torch.stack(channels_list, dim=0)
                    else:
                        mask_processed = channels_list[0].unsqueeze(0)
                    
                    # Add processed mask to batch list
                    masks_list.append(mask_processed)
                
                # Stack batch items back together
                if len(masks_list) > 1:
                    mask = torch.stack(masks_list, dim=0)
                else:
                    mask = masks_list[0].unsqueeze(0)
                
                # Convert back to proper binary format
                mask = mask.to(torch.uint8)
            
            # Remove batch dimension if the input wasn't batched
            if not is_batched and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            
            return mask
    except Exception as e:
        logger.error(f"Error creating HSV mask on GPU: {e}")
        # Fallback to simpler approach in case of error
        return None

def tensor_to_qimage(tensor):
    """
    Convert a tensor to a QImage for display.
    
    Args:
        tensor (torch.Tensor): Tensor to convert [H,W,C]
        
    Returns:
        QImage: QImage ready for display
    """
    if tensor is None:
        return None
        
    try:
        # Download to CPU if on GPU
        np_image = download_tensor(tensor)
        
        # Check if grayscale or RGB
        if len(np_image.shape) == 2 or np_image.shape[2] == 1:
            # Grayscale
            height, width = np_image.shape[:2]
            bytes_per_line = width
            return QImage(np_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # RGB
            height, width, channels = np_image.shape
            bytes_per_line = channels * width
            return QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    except Exception as e:
        logger.error(f"Error converting tensor to QImage: {e}")
        return None

def apply_mask_overlay_gpu(image_tensor, mask_tensor, alpha=0.3):
    """
    Apply a mask overlay on an image using GPU.
    
    Args:
        image_tensor (torch.Tensor): Original image tensor [H,W,3]
        mask_tensor (torch.Tensor): Binary mask tensor [H,W,1]
        alpha (float): Transparency factor
        
    Returns:
        torch.Tensor: Image with mask overlay
    """
    if image_tensor is None or mask_tensor is None:
        return image_tensor
        
    try:
        # Ensure tensors are on the correct device
        if image_tensor.device != DEVICE:
            image_tensor = image_tensor.to(DEVICE)
        if mask_tensor.device != DEVICE:
            mask_tensor = mask_tensor.to(DEVICE)
            
        # Create a red overlay (values in range [0,1])
        overlay = torch.zeros_like(image_tensor)
        # Set red channel (BGR format, so index 2)
        overlay[..., 2] = 1.0
        
        # Convert mask to binary float tensor [0.0, 1.0]
        binary_mask = (mask_tensor > 0).float()
        
        # Expand mask dimensions if needed
        if binary_mask.shape[-1] == 1 and image_tensor.shape[-1] == 3:
            binary_mask = binary_mask.repeat(1, 1, 3)
            
        # Apply overlay
        result = image_tensor * (1 - alpha * binary_mask) + overlay * (alpha * binary_mask)
        
        return result
    except Exception as e:
        logger.error(f"Error applying mask overlay on GPU: {e}")
        return image_tensor

def get_gpu_info():
    """
    Get information about GPU availability and memory usage.
    
    Returns:
        dict: GPU information
    """
    info = {
        "available": IS_GPU_AVAILABLE,
        "device": str(DEVICE),
        "name": "Unknown",
        "memory_allocated": 0,
        "memory_cached": 0,
        "memory_reserved": 0,
        "memory_total": 0
    }
    
    if IS_GPU_AVAILABLE:
        try:
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            info["memory_cached"] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            info["memory_reserved"] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            
            # Total memory requires pynvml, so we'll skip for now
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            
    return info 