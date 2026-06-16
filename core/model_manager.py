"""Model management module for loading and caching Qwen3-VL models."""

import gc
import os
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import MODELS, ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manages model loading, caching, and GPU memory."""
    
    def __init__(self):
        """Initialize model manager."""
        self._lock = threading.Lock()
        self._cache: dict[str, Tuple[Any, AutoProcessor]] = {}
    
    def _get_local_model_path(self, model_id: str) -> Optional[str]:
        """Check if model exists in local HuggingFace cache.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "Qwen/Qwen3-VL-2B-Instruct")
            
        Returns:
            Local model path if exists, None otherwise
        """
        # Convert model_id to cache directory format
        # "Qwen/Qwen3-VL-2B-Instruct" -> "models--Qwen--Qwen3-VL-2B-Instruct"
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = cache_dir / model_cache_name
        
        if not model_cache_path.exists():
            return None
        
        # Check for snapshots
        snapshots_dir = model_cache_path / "snapshots"
        if not snapshots_dir.exists():
            return None
        
        # Find the latest snapshot
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            return None
        
        # Get the snapshot with model files
        for snapshot in snapshots:
            if (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists():
                logger.info(f"Found local model: {model_cache_path}")
                return str(snapshot)
        
        return None
    
    def load_model(self, model_id: str) -> Tuple[Any, AutoProcessor]:
        """Load model with caching and automatic memory management.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Tuple of (model, processor)
            
        Raises:
            RuntimeError: If model loading fails
        """
        with self._lock:
            if model_id in self._cache:
                logger.info(f"Using cached model: {model_id}")
                return self._cache[model_id]
            
            # Release other models to free GPU memory
            self._release_other_models(model_id)
            
            # Check for local model
            local_path = self._get_local_model_path(model_id)
            
            if local_path:
                logger.info(f"Loading model from local cache: {local_path}")
                model_source = local_path
            else:
                logger.info(f"Model not found locally, will download from HuggingFace: {model_id}")
                model_source = model_id
            
            # Load processor
            logger.info(f"Loading processor for {model_id}...")
            processor = AutoProcessor.from_pretrained(model_source)
            
            # Load model with fallback for older transformers versions
            logger.info(f"Loading model {model_id}...")
            
            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                # MPS doesn't support device_map="auto" well, use cpu then move to mps
                device_map = None
                dtype = torch.float16
            else:
                device_map = None
                dtype = torch.float32
            
            try:
                if device_map:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_source,
                        dtype=dtype,
                        device_map=device_map,
                    )
                else:
                    # Always load on CPU first, then move to target device
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_source,
                        torch_dtype=dtype,
                    )
                    # Move to MPS if available, fallback to CPU on failure
                    if torch.backends.mps.is_available():
                        try:
                            model = model.to("mps")
                            logger.info("Model moved to MPS device")
                        except Exception as e:
                            logger.warning(f"MPS allocation failed ({e}), keeping on CPU")
                            # Model stays on CPU, no need to do anything
            except TypeError:
                if device_map:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_source,
                        dtype=dtype,
                        device_map=device_map,
                    )
                else:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_source,
                        dtype=dtype,
                    )
                    # Move to MPS if available, fallback to CPU on failure
                    if torch.backends.mps.is_available():
                        try:
                            model = model.to("mps")
                            logger.info("Model moved to MPS device")
                        except Exception as e:
                            logger.warning(f"MPS allocation failed ({e}), keeping on CPU")
            
            model.eval()
            
            self._cache[model_id] = (model, processor)
            
            # Log model info
            model_size = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info(f"Model loaded successfully: {model_id} ({model_size:.1f}M parameters)")
            
            return model, processor
    
    def _release_other_models(self, keep_model_id: str) -> None:
        """Release models other than the specified one.
        
        Args:
            keep_model_id: Model ID to keep in cache
        """
        keys_to_remove = [k for k in self._cache if k != keep_model_id]
        for k in keys_to_remove:
            logger.info(f"Releasing model: {k}")
            del self._cache[k]
        
        if keys_to_remove:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def release_all(self) -> None:
        """Release all models and free GPU memory."""
        with self._lock:
            self._cache.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "loaded_models": list(self._cache.keys()),
            "gpu_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "local_models": {},
        }
        
        # Check local models
        for model_key, model_config in MODELS.items():
            local_path = self._get_local_model_path(model_config.model_id)
            info["local_models"][model_key] = {
                "model_id": model_config.model_id,
                "local_path": local_path,
                "downloaded": local_path is not None,
            }
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        return info
    
    def decode_generation(
        self,
        processor: AutoProcessor,
        inputs: Any,
        generated_ids: Any
    ) -> str:
        """Decode generated token IDs to text.
        
        Args:
            processor: Model processor
            inputs: Model inputs
            generated_ids: Generated token IDs
            
        Returns:
            Decoded text
        """
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0].strip()


    def preload_model(self, model_id: str) -> None:
        """Preload model into cache (non-blocking hint).
        
        Args:
            model_id: HuggingFace model identifier
        """
        if model_id in self._cache:
            logger.info(f"Model already cached: {model_id}")
            return
        
        logger.info(f"Preloading model: {model_id}")
        try:
            self.load_model(model_id)
            logger.info(f"Model preloaded successfully: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to preload model {model_id}: {e}")
    
    def warmup(self, model_id: str) -> None:
        """Warm up model with a dummy inference pass.
        
        Args:
            model_id: HuggingFace model identifier
        """
        if model_id not in self._cache:
            return
        
        model, processor = self._cache[model_id]
        
        try:
            # Create a minimal dummy input
            dummy_text = "warmup"
            inputs = processor(text=dummy_text, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
            
            logger.info(f"Model warmed up: {model_id}")
        except Exception as e:
            logger.warning(f"Warmup failed for {model_id}: {e}")
    
    def unload_model(self, model_id: str) -> str:
        """Unload model from memory.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Status message
        """
        with self._lock:
            if model_id not in self._cache:
                return f"模型 {model_id} 未在内存中"
            
            del self._cache[model_id]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"Model unloaded: {model_id}")
        return f"模型 {model_id} 已从内存卸载"
    
    def delete_local_model(self, model_key: str) -> str:
        """Delete model from local HuggingFace cache.
        
        Args:
            model_key: Model key (e.g., "2B", "4B", "8B")
            
        Returns:
            Status message
        """
        if model_key not in MODELS:
            return f"未知模型: {model_key}"
        
        model_id = MODELS[model_key].model_id
        
        # First unload from memory if loaded
        if model_id in self._cache:
            self.unload_model(model_id)
        
        # Find and delete local cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = cache_dir / model_cache_name
        
        if not model_cache_path.exists():
            return f"模型 {model_key} 本地缓存不存在"
        
        try:
            import shutil
            shutil.rmtree(model_cache_path)
            logger.info(f"Deleted local model cache: {model_cache_path}")
            return f"模型 {model_key} 本地缓存已删除"
        except Exception as e:
            return f"删除失败: {e}"
    
    def get_model_disk_size(self, model_key: str) -> str:
        """Get model disk size.
        
        Args:
            model_key: Model key (e.g., "2B", "4B", "8B")
            
        Returns:
            Formatted size string
        """
        if model_key not in MODELS:
            return "未知"
        
        model_id = MODELS[model_key].model_id
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_name = f"models--{model_id.replace('/', '--')}"
        model_cache_path = cache_dir / model_cache_name
        
        if not model_cache_path.exists():
            return "未下载"
        
        try:
            total_size = sum(f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file())
            
            if total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                return f"{total_size / (1024 * 1024):.1f} MB"
            else:
                return f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        except Exception:
            return "计算失败"


# Global model manager instance
model_manager = ModelManager()
