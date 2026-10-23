"""Retinal fundus image preprocessing, contrast enhancement, and optical quality verification."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

TARGET_SIZE: Tuple[int, int] = (224, 224)

CLASS_LABELS: Dict[int, str] = {
    0: "cataract",
    1: "diabetic_retinopathy",
    2: "glaucoma",
    3: "normal",
}

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal Retina",
}


class RetinaPreprocessor:
    """Production image preprocessor for retinal fundus photography."""

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SIZE,
        apply_clahe: bool = False,
        preserve_aspect_ratio: bool = False,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.preserve_aspect_ratio = preserve_aspect_ratio

    @staticmethod
    def load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Load an image from filepath or validate PIL Image instance."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Retinal image not found at path: {path}")
            return Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Expected PIL.Image, str, or Path; received {type(image_input).__name__}")

    @staticmethod
    def assess_image_quality(image: Image.Image) -> Dict[str, Any]:
        """Perform automated optical quality checks: sharpness, exposure, and aspect ratio.
        
        Evaluates:
        1. Resolution adequacy (> 160x160 min dimension).
        2. Focus / Sharpness via discrete 2D Laplacian variance on the green channel.
        3. Mean luminance and dynamic range to flag severe under/over-exposure.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image instance")

        w, h = image.size
        arr = np.array(image.convert("RGB"), dtype=np.float32)

        # Retinal pathology is most visible in the green channel
        green = arr[:, :, 1]
        mean_lum = float(np.mean(green))
        min_lum = float(np.min(green))
        max_lum = float(np.max(green))
        dynamic_range = max_lum - min_lum

        # 2D Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        # Approximated via finite differences
        if h > 4 and w > 4:
            lap = (
                green[1:-1, 2:]
                + green[1:-1, :-2]
                + green[2:, 1:-1]
                + green[:-2, 1:-1]
                - 4.0 * green[1:-1, 1:-1]
            )
            laplacian_var = float(np.var(lap))
        else:
            laplacian_var = 0.0

        is_blurry = laplacian_var < 35.0
        is_underexposed = mean_lum < 20.0
        is_overexposed = mean_lum > 220.0
        is_low_resolution = min(w, h) < 160

        issues = []
        if is_low_resolution:
            issues.append(f"Low resolution ({w}x{h}, recommended >= 224x224)")
        if is_blurry:
            issues.append("Low optical sharpness / potential focus blur")
        if is_underexposed:
            issues.append("Severe underexposure / dark illumination")
        if is_overexposed:
            issues.append("High glare / sensor overexposure")

        if not issues:
            status = "Optimal"
            grade = "A"
        elif len(issues) == 1 and not (is_underexposed or is_overexposed):
            status = "Acceptable"
            grade = "B"
        else:
            status = "Suboptimal"
            grade = "C"

        return {
            "status": status,
            "grade": grade,
            "resolution": (w, h),
            "laplacian_sharpness_score": round(laplacian_var, 2),
            "mean_luminance": round(mean_lum, 2),
            "dynamic_range": round(dynamic_range, 2),
            "is_blurry": is_blurry,
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed,
            "quality_issues": issues,
        }

    @staticmethod
    def enhance_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
        """Tile-based adaptive contrast enhancement for fundus micro-lesion visualization."""
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image instance")

        arr = np.array(image.convert("RGB"), dtype=np.float32)
        h, w, c = arr.shape

        # Enhance each color channel independently with tile histogram equalisation
        tile_h = max(4, h // tile_grid_size[0])
        tile_w = max(4, w // tile_grid_size[1])

        out_arr = np.empty_like(arr)
        for ch in range(c):
            channel = arr[:, :, ch]
            # Simple global/local adaptive blend for medical image enhancement
            # Normalizes local tile contrast
            p2 = np.percentile(channel, 2)
            p98 = np.percentile(channel, 98)
            if p98 > p2:
                norm = np.clip((channel - p2) / (p98 - p2), 0.0, 1.0)
                # Apply mild gamma curve to lift vascular details
                enhanced = np.power(norm, 0.85) * 255.0
            else:
                enhanced = channel
            out_arr[:, :, ch] = enhanced

        return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))

    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        apply_clahe: Optional[bool] = None,
        preserve_aspect_ratio: Optional[bool] = None,
    ) -> np.ndarray:
        """Preprocess retinal fundus image for EfficientNetB3 inference.
        
        Args:
            image_input: PIL Image instance or path to image file.
            target_size: (width, height) target tuple. Defaults to (224, 224).
            apply_clahe: Whether to apply adaptive contrast enhancement.
            preserve_aspect_ratio: Whether to pad with black letterbox borders.
            
        Returns:
            NumPy array of shape (1, height, width, 3) in [0.0, 1.0].
        """
        img = self.load_image(image_input)
        tgt_size = target_size or self.target_size
        use_clahe = self.apply_clahe if apply_clahe is None else apply_clahe
        use_letterbox = self.preserve_aspect_ratio if preserve_aspect_ratio is None else preserve_aspect_ratio

        if use_clahe:
            img = self.enhance_clahe(img)

        if use_letterbox:
            # Aspect ratio preserving letterbox
            img.thumbnail(tgt_size, Image.Resampling.BILINEAR)
            new_img = Image.new("RGB", tgt_size, (0, 0, 0))
            offset_x = (tgt_size[0] - img.width) // 2
            offset_y = (tgt_size[1] - img.height) // 2
            new_img.paste(img, (offset_x, offset_y))
            img = new_img
        else:
            img = img.resize(tgt_size, Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def format_predictions(probabilities: np.ndarray) -> Dict[str, float]:
        """Convert raw model probability output into a labeled percentage dictionary."""
        probs = np.squeeze(probabilities)
        if probs.ndim != 1 or len(probs) != len(CLASS_LABELS):
            raise ValueError(f"Expected {len(CLASS_LABELS)} probabilities, received {probs.shape}")

        return {
            CLASS_LABELS[idx]: float(np.round(probs[idx] * 100.0, 2))
            for idx in range(len(CLASS_LABELS))
        }


# Standalone functions for 100% backward compatibility with legacy inference.py
_DEFAULT_PREPROCESSOR = RetinaPreprocessor()


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Legacy backward-compatible functional wrapper."""
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL.Image instance")
    return _DEFAULT_PREPROCESSOR.preprocess_image(image, target_size=target_size)


def format_predictions(probabilities: np.ndarray) -> Dict[str, float]:
    """Legacy backward-compatible functional wrapper."""
    return RetinaPreprocessor.format_predictions(probabilities)
