"""DGX Spark GPU configuration and optimization utilities."""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration settings for DGX Spark."""

    memory_fraction: float = 0.95
    batch_size_auto: bool = True
    default_batch_size: int = 32
    max_workers: int = 8
    use_amp: bool = True
    use_bf16: bool = True
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    torch_compile_mode: str = "max-autotune"
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_env(cls) -> "GPUConfig":
        """Load GPU configuration from environment variables."""
        return cls(
            memory_fraction=float(os.getenv("GPU_MEMORY_FRACTION", "0.95")),
            batch_size_auto=os.getenv("BATCH_SIZE_AUTO", "true").lower() == "true",
            default_batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "32")),
            max_workers=int(os.getenv("MAX_WORKERS", "8")),
            use_amp=os.getenv("USE_AMP", "true").lower() == "true",
            use_bf16=os.getenv("USE_BF16", "true").lower() == "true",
            use_flash_attention=os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true",
            use_torch_compile=os.getenv("USE_TORCH_COMPILE", "true").lower() == "true",
            torch_compile_mode=os.getenv("TORCH_COMPILE_MODE", "max-autotune"),
            use_gradient_checkpointing=os.getenv("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true",
        )


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Get GPU information if CUDA is available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "device_name": props.name,
            "device_index": device,
            "total_memory_gb": props.total_memory / (1024**3),
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
        }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def configure_gpu(config: Optional[GPUConfig] = None) -> Dict[str, Any]:
    """
    Configure GPU settings optimized for DGX Spark.

    Args:
        config: GPU configuration settings. If None, loads from environment.

    Returns:
        Dictionary containing GPU configuration status.
    """
    if config is None:
        config = GPUConfig.from_env()

    result = {
        "cuda_available": False,
        "gpu_configured": False,
        "config_applied": {},
        "errors": [],
    }

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU.")
            return result

        result["cuda_available"] = True

        # Set default GPU device
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Clear GPU cache
        torch.cuda.empty_cache()
        result["config_applied"]["cache_cleared"] = True

        # Enable TF32 for Blackwell architecture (significant speedup)
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            result["config_applied"]["tf32_enabled"] = True
            logger.info("TF32 enabled for matrix operations")

        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True
        result["config_applied"]["cudnn_benchmark"] = True
        logger.info("cuDNN benchmark mode enabled")

        # Set memory fraction (for memory management)
        if config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(config.memory_fraction)
            result["config_applied"]["memory_fraction"] = config.memory_fraction
            logger.info(f"GPU memory fraction set to {config.memory_fraction}")

        # Log GPU information
        gpu_info = get_gpu_info()
        if gpu_info:
            logger.info(
                f"GPU: {gpu_info['device_name']}, "
                f"Memory: {gpu_info['total_memory_gb']:.1f}GB, "
                f"CUDA: {gpu_info['cuda_version']}"
            )
            result["gpu_info"] = gpu_info

        result["gpu_configured"] = True

    except ImportError:
        result["errors"].append("PyTorch not installed")
        logger.error("PyTorch is not installed")
    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"GPU configuration failed: {e}")

    return result


def get_optimal_batch_size(
    model_name: str,
    sequence_length: int = 512,
    base_batch_size: int = 8,
    config: Optional[GPUConfig] = None,
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Args:
        model_name: Name of the model being used.
        sequence_length: Maximum sequence length.
        base_batch_size: Base batch size to scale from.
        config: GPU configuration settings.

    Returns:
        Recommended batch size.
    """
    if config is None:
        config = GPUConfig.from_env()

    if not config.batch_size_auto:
        return config.default_batch_size

    try:
        import torch

        if not torch.cuda.is_available():
            return base_batch_size

        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        available_memory = (total_memory - allocated_memory) * config.memory_fraction

        # Estimate memory per sample (rough heuristic)
        # This is a simplified calculation and should be tuned per model
        model_multipliers = {
            "scibert": 1.0,
            "bert": 1.0,
            "roberta": 1.1,
            "allenai/scibert": 1.0,
            "allenai/scibert_scivocab_uncased": 1.0,
        }

        multiplier = model_multipliers.get(model_name.lower(), 1.0)

        # Base memory estimation (MB per sample)
        base_memory_per_sample = 50 * multiplier * (sequence_length / 512)

        # Calculate optimal batch size
        optimal_batch_size = int(
            (available_memory / (1024**2)) / base_memory_per_sample
        )

        # Clamp to reasonable range
        optimal_batch_size = max(1, min(optimal_batch_size, 128))

        # Round down to power of 2 for efficiency
        optimal_batch_size = 2 ** (optimal_batch_size.bit_length() - 1)

        logger.info(
            f"Optimal batch size for {model_name}: {optimal_batch_size} "
            f"(available memory: {available_memory / (1024**3):.1f}GB)"
        )

        return optimal_batch_size

    except Exception as e:
        logger.warning(f"Failed to calculate optimal batch size: {e}")
        return config.default_batch_size


def setup_mixed_precision(config: Optional[GPUConfig] = None) -> Optional[Any]:
    """
    Set up automatic mixed precision (AMP) for faster inference.

    Args:
        config: GPU configuration settings.

    Returns:
        Autocast context manager or None if not available.
    """
    if config is None:
        config = GPUConfig.from_env()

    try:
        import torch
        from torch.cuda.amp import autocast

        if not torch.cuda.is_available() or not config.use_amp:
            return None

        dtype = torch.bfloat16 if config.use_bf16 else torch.float16

        logger.info(f"Mixed precision enabled with dtype: {dtype}")

        return autocast(dtype=dtype)

    except Exception as e:
        logger.warning(f"Failed to setup mixed precision: {e}")
        return None


def compile_model(model: Any, config: Optional[GPUConfig] = None) -> Any:
    """
    Compile model using torch.compile for optimized inference.

    Args:
        model: PyTorch model to compile.
        config: GPU configuration settings.

    Returns:
        Compiled model or original model if compilation fails.
    """
    if config is None:
        config = GPUConfig.from_env()

    if not config.use_torch_compile:
        return model

    try:
        import torch

        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model

        compiled_model = torch.compile(model, mode=config.torch_compile_mode)
        logger.info(f"Model compiled with mode: {config.torch_compile_mode}")

        return compiled_model

    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")
        return model


def enable_gradient_checkpointing(model: Any, config: Optional[GPUConfig] = None) -> Any:
    """
    Enable gradient checkpointing to reduce memory usage.

    Args:
        model: PyTorch model.
        config: GPU configuration settings.

    Returns:
        Model with gradient checkpointing enabled.
    """
    if config is None:
        config = GPUConfig.from_env()

    if not config.use_gradient_checkpointing:
        return model

    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")

        return model

    except Exception as e:
        logger.warning(f"Failed to enable gradient checkpointing: {e}")
        return model


# Initialize GPU configuration on module import
def initialize_gpu() -> Dict[str, Any]:
    """Initialize GPU configuration on startup."""
    config = GPUConfig.from_env()
    return configure_gpu(config)


# Global GPU configuration instance
gpu_config = GPUConfig.from_env()
