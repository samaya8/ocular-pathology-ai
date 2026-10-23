"""Diagnostic uncertainty quantification engine using Monte Carlo Dropout and Shannon entropy."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.preprocessor import CLASS_LABELS, CLASS_DISPLAY_NAMES


class UncertaintyEngine:
    """Estimates epistemic and aleatoric uncertainty via Monte Carlo Dropout."""

    def __init__(self, num_samples: int = 20) -> None:
        self.num_samples = max(2, int(num_samples))

    def evaluate_uncertainty(
        self,
        model: Any,
        preprocessed_img: np.ndarray,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform Monte Carlo Dropout stochastic forward passes to assess diagnostic confidence.
        
        Args:
            model: Keras or Mock model instance.
            preprocessed_img: Array of shape (1, H, W, 3).
            num_samples: Optional override for number of stochastic passes.
            
        Returns:
            Dictionary containing predictive mean, epistemic variance, entropy, and reliability metrics.
        """
        n_passes = num_samples or self.num_samples
        samples: List[np.ndarray] = []

        # Check if model has standard EfficientNet multi-layer structure
        is_composite_keras = (
            hasattr(model, "layers")
            and len(model.layers) >= 5
            and hasattr(model.layers[0], "__call__")
        )

        if is_composite_keras:
            try:
                import tensorflow as tf
                img_t = tf.constant(preprocessed_img, dtype=tf.float32)
                base_model = model.layers[0]
                features = base_model(img_t, training=False)
                bn_layer = model.layers[1]
                dense2_layer = model.layers[2]
                drop_layer = model.layers[3]
                out_layer = model.layers[4]

                # Pre-compute BN and Dense2 before stochastic dropout
                h = dense2_layer(bn_layer(features, training=False))
                for _ in range(n_passes):
                    # Stochastic Dropout pass
                    d_out = drop_layer(h, training=True)
                    logits = out_layer(d_out)
                    samples.append(np.squeeze(logits.numpy()))
            except Exception:
                samples = []

        if not samples:
            # Fallback for MockKerasModel or arbitrary estimators
            for _ in range(n_passes):
                preds = model.predict(preprocessed_img, verbose=0, training=True)
                p = np.squeeze(np.asarray(preds))
                samples.append(p)

        samples_arr = np.array(samples, dtype=np.float32)  # Shape: (N, 4)

        # 1. Predictive Mean & Standard Deviation
        mean_probs = np.mean(samples_arr, axis=0)
        # Ensure exact probability simplex
        mean_probs = mean_probs / np.sum(mean_probs)
        std_probs = np.std(samples_arr, axis=0)

        # 2. Predictive Entropy (in bits, base 2)
        eps = 1e-8
        pred_entropy = float(-np.sum(mean_probs * np.log2(mean_probs + eps)))
        # Max entropy for 4 classes is log2(4) = 2.0 bits
        max_entropy = float(np.log2(len(CLASS_LABELS)))
        norm_entropy = float(np.clip(pred_entropy / max_entropy, 0.0, 1.0))

        # 3. Expected Sample Entropy (Aleatoric component)
        sample_entropies = -np.sum(samples_arr * np.log2(samples_arr + eps), axis=1)
        expected_sample_entropy = float(np.mean(sample_entropies))

        # 4. Mutual Information (Epistemic component)
        # I(y, W) = H(y|x) - E_{p(W)}[H(y|x, W)]
        mutual_info = float(max(0.0, pred_entropy - expected_sample_entropy))

        # 5. Class breakdown with 95% Credible Intervals
        class_breakdown: Dict[str, Dict[str, float]] = {}
        for idx in range(len(CLASS_LABELS)):
            label = CLASS_LABELS[idx]
            display = CLASS_DISPLAY_NAMES[label]
            m_val = float(mean_probs[idx])
            s_val = float(std_probs[idx])
            ci_low = float(np.clip(m_val - 1.96 * s_val, 0.0, 1.0))
            ci_high = float(np.clip(m_val + 1.96 * s_val, 0.0, 1.0))
            class_breakdown[label] = {
                "class_name": display,
                "mean_probability": round(m_val, 4),
                "mean_percentage": round(m_val * 100.0, 2),
                "epistemic_std": round(s_val, 4),
                "ci_95_low": round(ci_low, 4),
                "ci_95_high": round(ci_high, 4),
            }

        top_class_idx = int(np.argmax(mean_probs))
        top_class_label = CLASS_LABELS[top_class_idx]
        top_confidence = float(mean_probs[top_class_idx])
        top_std = float(std_probs[top_class_idx])

        # 6. Qualitative Uncertainty Tier & Reliability Scoring
        if norm_entropy < 0.35 and top_std < 0.06:
            uncertainty_tier = "Low Uncertainty (High Clinical Confidence)"
            second_opinion_recommended = False
        elif norm_entropy < 0.65 and top_std < 0.12:
            uncertainty_tier = "Moderate Uncertainty (Standard Review)"
            second_opinion_recommended = False
        else:
            uncertainty_tier = "High Ambiguity (Second Specialist Review Mandatory)"
            second_opinion_recommended = True

        # Diagnostic Reliability Index (0–100%)
        reliability_score = float(np.clip(
            100.0 * (1.0 - norm_entropy) * (1.0 - min(1.0, 2.5 * top_std)),
            5.0,
            99.0,
        ))

        return {
            "num_mc_samples": n_passes,
            "predicted_class": top_class_label,
            "predicted_display_name": CLASS_DISPLAY_NAMES[top_class_label],
            "confidence_percentage": round(top_confidence * 100.0, 2),
            "epistemic_uncertainty_std": round(top_std, 4),
            "shannon_entropy_bits": round(pred_entropy, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "mutual_information": round(mutual_info, 4),
            "uncertainty_tier": uncertainty_tier,
            "reliability_score": round(reliability_score, 1),
            "second_opinion_recommended": second_opinion_recommended,
            "class_breakdown": class_breakdown,
            "mc_samples_matrix": samples_arr.tolist(),
        }
