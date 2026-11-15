"""
Mock Data & Deterministic Demo Simulation Module
------------------------------------------------
This module isolates all mock datasets, sample image generators, mock model
predictions, and mock visual effects so they can be cleanly replaced with real
PyTorch model checkpoints and FGSM/PGD implementations later.

DEMO MODE ONLY: No real neural networks, gradients, or defenses are executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    dataset: str
    architecture: str
    input_size: str
    classes: List[str]
    clean_accuracy: str


@dataclass(frozen=True)
class SampleSpec:
    id: str
    label: str
    dataset: str
    model_id: str
    true_class: str
    target_misclass: str
    secondary_misclass: str
    base_confidence: float


@dataclass(frozen=True)
class PredictionState:
    predicted_class: str
    confidence: float
    is_correct: Optional[bool]
    status_tag: str
    metadata_lines: List[str]


@dataclass(frozen=True)
class MockAnalysisResult:
    original_image: Image.Image
    attacked_image: Image.Image
    defended_image: Image.Image
    original_pred: PredictionState
    attacked_pred: PredictionState
    defended_pred: PredictionState
    summary_headline: str
    summary_explanation: str
    perturbation_l_inf: float


MODELS: Dict[str, ModelSpec] = {
    "mnist_cnn": ModelSpec(
        id="mnist_cnn",
        name="MNIST / SimpleCNN",
        dataset="MNIST (28x28 Grayscale)",
        architecture="2-Layer ConvNet (Mock)",
        input_size="28 x 28",
        classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        clean_accuracy="99.1% (Mock)",
    ),
    "cifar10_resnet18": ModelSpec(
        id="cifar10_resnet18",
        name="CIFAR-10 / ResNet-18",
        dataset="CIFAR-10 (32x32 RGB)",
        architecture="ResNet-18 (Mock)",
        input_size="32 x 32",
        classes=[
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ],
        clean_accuracy="94.6% (Mock)",
    ),
    "gtsrb_resnet18": ModelSpec(
        id="gtsrb_resnet18",
        name="GTSRB / ResNet-18",
        dataset="GTSRB Traffic Signs",
        architecture="ResNet-18 (Mock)",
        input_size="32 x 32",
        classes=[
            "Stop Sign",
            "Speed Limit 50",
            "Yield",
            "No Entry",
            "Turn Right Ahead",
            "Priority Road",
        ],
        clean_accuracy="97.8% (Mock)",
    ),
}

SAMPLES: Dict[str, SampleSpec] = {
    "sample_cifar_plane": SampleSpec(
        id="sample_cifar_plane",
        label="CIFAR-10 · Airplane",
        dataset="CIFAR-10",
        model_id="cifar10_resnet18",
        true_class="Airplane",
        target_misclass="Bird",
        secondary_misclass="Ship",
        base_confidence=0.964,
    ),
    "sample_gtsrb_stop": SampleSpec(
        id="sample_gtsrb_stop",
        label="GTSRB · Stop Sign",
        dataset="GTSRB",
        model_id="gtsrb_resnet18",
        true_class="Stop Sign",
        target_misclass="Speed Limit 50",
        secondary_misclass="Yield",
        base_confidence=0.982,
    ),
    "sample_mnist_seven": SampleSpec(
        id="sample_mnist_seven",
        label="MNIST · Digit 7",
        dataset="MNIST",
        model_id="mnist_cnn",
        true_class="7",
        target_misclass="1",
        secondary_misclass="9",
        base_confidence=0.993,
    ),
    "sample_cifar_frog": SampleSpec(
        id="sample_cifar_frog",
        label="CIFAR-10 · Frog",
        dataset="CIFAR-10",
        model_id="cifar10_resnet18",
        true_class="Frog",
        target_misclass="Cat",
        secondary_misclass="Deer",
        base_confidence=0.938,
    ),
}

ATTACK_METHODS: List[str] = ["FGSM", "PGD"]
DEFENSE_METHODS: List[str] = [
    "None",
    "PGD-AT",
    "Gaussian Blur",
    "Bit-depth Reduction",
]


def generate_mock_sample_image(sample_id: str, size: int = 360) -> Image.Image:
    """Create crisp synthetic demo images with Pillow so no external files are required."""
    img = Image.new("RGB", (size, size), color=(245, 243, 238))
    draw = ImageDraw.Draw(img)

    if sample_id == "sample_mnist_seven":
        img = Image.new("RGB", (size, size), color=(18, 18, 22))
        draw = ImageDraw.Draw(img)
        # Draw bold handwritten-style digit "7"
        draw.line(
            [(85, 92), (275, 92), (145, 295)],
            fill=(248, 248, 250),
            width=36,
            joint="curve",
        )
        draw.line([(155, 192), (242, 192)], fill=(230, 230, 236), width=18)
        # Subtle pixel-grid feel
        small = img.resize((56, 56), Image.Resampling.BILINEAR)
        return small.resize((size, size), Image.Resampling.NEAREST)

    if sample_id == "sample_gtsrb_stop":
        # Sky & road backdrop
        draw.rectangle([0, 0, size, int(size * 0.65)], fill=(138, 196, 232))
        draw.rectangle([0, int(size * 0.65), size, size], fill=(92, 96, 104))
        # Post
        draw.rectangle([size // 2 - 10, int(size * 0.65), size // 2 + 10, size], fill=(60, 62, 68))
        # Octagon STOP sign
        cx, cy, r = size // 2, int(size * 0.44), 108
        angles = [np.pi / 8 + i * np.pi / 4 for i in range(8)]
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        draw.polygon(pts, fill=(214, 38, 38), outline=(255, 255, 255))
        r_in = 96
        pts_in = [(cx + r_in * np.cos(a), cy + r_in * np.sin(a)) for a in angles]
        draw.polygon(pts_in, fill=(214, 38, 38), outline=(255, 255, 255))
        font = ImageFont.load_default()
        draw.text((cx - 32, cy - 10), "STOP", fill=(255, 255, 255), font=font)
        return img

    if sample_id == "sample_cifar_frog":
        # Leaf green background
        draw.rectangle([0, 0, size, size], fill=(62, 118, 66))
        draw.ellipse([30, 190, 330, 335], fill=(42, 86, 48))
        # Frog body
        draw.ellipse([95, 130, 265, 270], fill=(108, 178, 72), outline=(25, 45, 22), width=4)
        # Eyes
        draw.ellipse([112, 95, 162, 145], fill=(125, 196, 82), outline=(25, 45, 22), width=4)
        draw.ellipse([198, 95, 248, 145], fill=(125, 196, 82), outline=(25, 45, 22), width=4)
        draw.ellipse([128, 110, 146, 128], fill=(15, 15, 18))
        draw.ellipse([214, 110, 232, 128], fill=(15, 15, 18))
        draw.arc([135, 165, 225, 210], start=10, end=170, fill=(25, 45, 22), width=4)
        return img

    # Default: CIFAR-10 Airplane
    draw.rectangle([0, 0, size, int(size * 0.72)], fill=(96, 168, 226))
    draw.rectangle([0, int(size * 0.72), size, size], fill=(224, 234, 244))
    # Clouds
    draw.ellipse([32, 54, 128, 98], fill=(248, 250, 252))
    draw.ellipse([235, 42, 326, 86], fill=(248, 250, 252))
    # Airplane silhouette
    draw.polygon([(64, 192), (288, 166), (308, 178), (278, 196), (78, 208)], fill=(242, 244, 248), outline=(24, 28, 36))
    draw.polygon([(152, 182), (206, 98), (228, 98), (192, 180)], fill=(214, 220, 230), outline=(24, 28, 36))
    draw.polygon([(156, 196), (208, 268), (228, 268), (192, 194)], fill=(202, 210, 222), outline=(24, 28, 36))
    draw.polygon([(74, 192), (56, 142), (82, 142), (104, 186)], fill=(220, 58, 48), outline=(24, 28, 36))
    return img


def _apply_mock_perturbation(
    image: Image.Image,
    epsilon: float,
    attack_method: str,
    amplification: float = 1.0,
) -> Image.Image:
    """Apply deterministic visual noise to illustrate adversarial perturbation."""
    if epsilon <= 0.0:
        return image.copy()

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    if attack_method == "FGSM":
        # Signed high-frequency stripe + checkerboard pattern
        pattern = np.sign(np.sin(xx * 0.65) + np.cos(yy * 0.65))
        noise = np.stack([pattern, -pattern * 0.7, pattern * 0.85], axis=-1)
    else:
        # PGD multi-frequency structured perturbation
        p1 = np.sin(xx * 0.42 + yy * 0.31)
        p2 = np.cos(xx * 0.83 - yy * 0.57)
        p3 = np.sin((xx + yy) * 0.25)
        noise = np.stack([p1 + 0.5 * p2, p2 - 0.4 * p3, p3 + 0.5 * p1], axis=-1)
        noise = np.clip(noise, -1.0, 1.0)

    visual_scale = min(0.65, epsilon * amplification * 1.15)
    perturbed = np.clip(arr + noise * visual_scale, 0.0, 1.0)
    return Image.fromarray((perturbed * 255.0).astype(np.uint8))


def _apply_mock_defense_visual(
    attacked_image: Image.Image,
    original_image: Image.Image,
    defense_method: str,
    epsilon: float,
    blur_radius: float,
    bit_depth: int,
) -> Image.Image:
    """Apply purely visual mock defense transformation to the attacked preview."""
    if defense_method == "None":
        return attacked_image.copy()

    if defense_method == "PGD-AT":
        # PGD-AT evaluates a robust checkpoint on the attacked input.
        # Visually show a mildly denoised / saliently preserved view of the attacked sample.
        return attacked_image.filter(ImageFilter.SMOOTH_MORE)

    if defense_method == "Gaussian Blur":
        return attacked_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if defense_method == "Bit-depth Reduction":
        levels = max(2, (1 << bit_depth) - 1)
        arr = np.asarray(attacked_image.convert("RGB"), dtype=np.float32) / 255.0
        quantized = np.round(arr * levels) / levels
        return Image.fromarray((np.clip(quantized, 0.0, 1.0) * 255.0).astype(np.uint8))

    return attacked_image.copy()


def compute_mock_analysis(
    model_id: str,
    attack_method: str,
    epsilon: float,
    defense_method: str,
    sample_id: Optional[str] = None,
    uploaded_image: Optional[Image.Image] = None,
    pgd_steps: int = 10,
    pgd_alpha: float = 0.01,
    pgd_restarts: int = 1,
    blur_radius: float = 1.8,
    bit_depth: int = 3,
    visual_amp: float = 1.0,
) -> MockAnalysisResult:
    """
    Compute deterministic mock predictions, confidence scores, and explanation text.
    """
    model = MODELS[model_id]
    sample = SAMPLES.get(sample_id or "", None)

    if uploaded_image is not None:
        base_image = uploaded_image.convert("RGB").resize((360, 360), Image.Resampling.BILINEAR)
        # Derive deterministic mock class from uploaded image average brightness/channel dominance
        arr = np.asarray(base_image, dtype=np.float32)
        idx = int(arr.mean()) % len(model.classes)
        true_class = model.classes[idx]
        target_misclass = model.classes[(idx + 2) % len(model.classes)]
        secondary_misclass = model.classes[(idx + 5) % len(model.classes)]
        base_conf = 0.892 + (int(arr[:, :, 0].mean()) % 70) / 1000.0
        has_ground_truth = False
    else:
        assert sample is not None
        base_image = generate_mock_sample_image(sample.id, size=360)
        true_class = sample.true_class
        # If user switched model manually away from the sample's native dataset, map deterministically
        if true_class not in model.classes:
            idx = sum(ord(c) for c in sample.id) % len(model.classes)
            true_class = model.classes[idx]
            target_misclass = model.classes[(idx + 1) % len(model.classes)]
            secondary_misclass = model.classes[(idx + 3) % len(model.classes)]
            base_conf = 0.845
            has_ground_truth = False
        else:
            target_misclass = sample.target_misclass
            secondary_misclass = sample.secondary_misclass
            base_conf = sample.base_confidence
            has_ground_truth = True

    attacked_image = _apply_mock_perturbation(base_image, epsilon, attack_method, amplification=visual_amp)
    defended_image = _apply_mock_defense_visual(
        attacked_image, base_image, defense_method, epsilon, blur_radius, bit_depth
    )

    # 1. Original prediction
    orig_pred = PredictionState(
        predicted_class=true_class,
        confidence=round(base_conf, 3),
        is_correct=True if has_ground_truth else None,
        status_tag="CLEAN INPUT (MOCK)",
        metadata_lines=[
            f"Model: {model.name}",
            f"Dataset: {model.dataset}",
            "Perturbation: ε = 0.00",
        ],
    )

    # 2. Attacked prediction (PGD fools at lower epsilon than FGSM)
    attack_strength = epsilon * (1.35 + min(0.25, pgd_steps * 0.01) if attack_method == "PGD" else 1.0)
    flip_threshold = 0.05 if attack_method == "PGD" else 0.08

    if epsilon <= 0.005:
        atk_class = true_class
        atk_conf = base_conf
        atk_status = "NO PERTURBATION"
    elif attack_strength < flip_threshold:
        atk_class = true_class
        atk_conf = max(0.51, base_conf - attack_strength * 3.8)
        atk_status = "CONFIDENCE DEGRADED (MOCK)"
    elif attack_strength < 0.18:
        atk_class = target_misclass
        atk_conf = min(0.94, 0.56 + attack_strength * 2.1)
        atk_status = "ATTACK SUCCEEDED (MOCK)"
    else:
        atk_class = secondary_misclass
        atk_conf = min(0.985, 0.72 + attack_strength * 0.85)
        atk_status = "HIGH-CONFIDENCE FOOL (MOCK)"

    atk_meta = [f"Attack: {attack_method}", f"Epsilon (ε): {epsilon:.2f}"]
    if attack_method == "PGD":
        atk_meta.append(f"Steps: {pgd_steps} · α={pgd_alpha:.3f} · Restarts: {pgd_restarts}")
    else:
        atk_meta.append("Single-step L∞ gradient sign (Mock)")

    atk_pred = PredictionState(
        predicted_class=atk_class,
        confidence=round(atk_conf, 3),
        is_correct=(atk_class == true_class) if has_ground_truth else None,
        status_tag=atk_status,
        metadata_lines=atk_meta,
    )

    # 3. Defended prediction
    if defense_method == "None":
        def_class = atk_class
        def_conf = atk_conf
        def_status = "UNDEFENDED (PASS-THROUGH)"
        def_meta = ["Defense: None", "Input passed directly to standard model"]
    elif defense_method == "PGD-AT":
        # Robust model checkpoint resists up to ~0.16 mock epsilon
        if attack_strength < 0.24:
            def_class = true_class
            def_conf = max(0.68, base_conf - 0.06 - attack_strength * 0.65)
            def_status = "RECOVERED BY ROBUST MODEL (MOCK)"
        else:
            def_class = target_misclass
            def_conf = min(0.79, 0.48 + attack_strength * 0.7)
            def_status = "OVERWHELMED AT HIGH ε (MOCK)"
        def_meta = [
            "Defense: PGD-AT (Adversarial Training)",
            "Evaluated on robust checkpoint (Mock)",
        ]
    elif defense_method == "Gaussian Blur":
        # Moderate blur suppresses high-frequency noise, but too much blur or high eps degrades accuracy
        if epsilon <= 0.12 and 0.8 <= blur_radius <= 2.6:
            def_class = true_class
            def_conf = max(0.58, base_conf - 0.14 - epsilon * 1.1)
            def_status = "NOISE SMOOTHED (MOCK)"
        elif epsilon <= 0.05:
            def_class = true_class
            def_conf = max(0.55, base_conf - 0.12)
            def_status = "RECOVERED (MOCK)"
        else:
            def_class = secondary_misclass if blur_radius > 2.6 else target_misclass
            def_conf = min(0.82, 0.44 + epsilon * 1.1)
            def_status = "PARTIAL RECOVERY / BLUR LOSS (MOCK)"
        def_meta = [
            "Defense: Gaussian Blur (Input Transform)",
            f"Kernel σ = {blur_radius:.1f}px",
        ]
    else:  # Bit-depth Reduction
        if epsilon <= 0.14 and 2 <= bit_depth <= 4:
            def_class = true_class
            def_conf = max(0.61, base_conf - 0.11 - epsilon * 0.95)
            def_status = "QUANTIZATION RECOVERED (MOCK)"
        else:
            def_class = target_misclass
            def_conf = min(0.84, 0.49 + epsilon * 1.05)
            def_status = "QUANTIZATION INSUFFICIENT (MOCK)"
        def_meta = [
            "Defense: Bit-depth Reduction (Input Transform)",
            f"Precision: {bit_depth}-bit ({1 << bit_depth} levels/channel)",
        ]

    def_pred = PredictionState(
        predicted_class=def_class,
        confidence=round(def_conf, 3),
        is_correct=(def_class == true_class) if has_ground_truth else None,
        status_tag=def_status,
        metadata_lines=def_meta,
    )

    # 4. Summary text
    fooled = atk_class != true_class
    recovered = def_class == true_class

    if epsilon <= 0.005:
        headline = "Baseline Check: Zero Perturbation Applied"
        explanation = (
            f"With ε = 0.00, no adversarial perturbation is added to the {true_class} input. "
            f"Both Original and Attacked panels report '{true_class}' ({orig_pred.confidence:.1%})."
        )
    elif fooled and defense_method == "None":
        headline = f"{attack_method} Fooled {model.name}: {true_class} → {atk_class}"
        explanation = (
            f"At ε = {epsilon:.2f}, the mock {attack_method} perturbation shifted the model's decision "
            f"from '{true_class}' ({orig_pred.confidence:.1%}) to '{atk_class}' ({atk_pred.confidence:.1%}). "
            f"Because Defense is set to None, the Defended panel mirrors the unmitigated attack."
        )
    elif fooled and recovered:
        headline = f"{defense_method} Restored '{true_class}' After {attack_method} Attack"
        explanation = (
            f"The mock {attack_method} attack (ε = {epsilon:.2f}) fooled the standard classifier into predicting "
            f"'{atk_class}' ({atk_pred.confidence:.1%}). Applying {defense_method} recovered the prediction back to "
            f"'{def_class}' ({def_pred.confidence:.1%}) in this simulated demo scenario."
        )
    elif fooled and not recovered:
        headline = f"{defense_method} Failed to Fully Recover at ε = {epsilon:.2f}"
        explanation = (
            f"Under a strong mock {attack_method} perturbation (ε = {epsilon:.2f}), the prediction flipped from "
            f"'{true_class}' to '{atk_class}'. {defense_method} altered the confidence to {def_pred.confidence:.1%}, "
            f"but could not restore the original class '{true_class}'."
        )
    else:
        headline = f"Model Held Prediction '{true_class}' Under Light {attack_method} (ε = {epsilon:.2f})"
        explanation = (
            f"At ε = {epsilon:.2f}, the perturbation reduced confidence from {orig_pred.confidence:.1%} to "
            f"{atk_pred.confidence:.1%} without flipping the top class. Applying {defense_method} keeps the output "
            f"at '{def_class}' ({def_pred.confidence:.1%})."
        )

    return MockAnalysisResult(
        original_image=base_image,
        attacked_image=attacked_image,
        defended_image=defended_image,
        original_pred=orig_pred,
        attacked_pred=atk_pred,
        defended_pred=def_pred,
        summary_headline=headline,
        summary_explanation=explanation,
        perturbation_l_inf=epsilon,
    )
