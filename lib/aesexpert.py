"""AesExpert model for analyzing aesthetic experience of images."""

import os
# Disable tokenizers parallelism to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Union, Dict, Tuple
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Lazy load model
_model = None
_tokenizer = None


def build_transform(input_size):
    """Build image transform pipeline."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into multiple blocks."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """Load and preprocess an image."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_model():
    """Lazy load the AesExpert model."""
    global _model, _tokenizer

    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        model_path = 'HumanBeauty/HumanAesExpert-1B'

        print("Loading HumanAesExpert model (this may take a moment)...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            _model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            ).eval().cuda()
        else:
            # CPU fallback
            _model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        print("Model loaded successfully!")

    return _model, _tokenizer


def score_image(
    image: Union[str, Path, Image.Image],
    question: str = "Rate the aesthetics of this picture.",
    fast: bool = True
) -> float:
    """
    Score the aesthetic quality of an image.

    Args:
        image: Path to image file or PIL Image
        question: Question to ask about the image
        fast: If True, use fast inference (1x time). If False, use metavoter (2x time, more accurate)

    Returns:
        float: Aesthetic score

    Example:
        >>> score_image("design.png")
        7.5
    """
    model, tokenizer = get_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Load and preprocess image
    pixel_values = load_image(image, max_num=12).to(dtype)
    if device == 'cuda':
        pixel_values = pixel_values.cuda()

    question_with_image = f'<image>\n{question}'

    if fast:
        # Fast inference
        pred_score = model.score(tokenizer, pixel_values, question_with_image)
    else:
        # Slow inference with metavoter
        pred_score = model.run_metavoter(tokenizer, pixel_values)

    return float(pred_score)


def get_expert_scores(image: Union[str, Path, Image.Image]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Get detailed expert scores across 12 aesthetic dimensions.

    Args:
        image: Path to image file or PIL Image

    Returns:
        Tuple of (expert_scores dict, expert_text dict) with 12 dimensions

    Example:
        >>> scores, texts = get_expert_scores("design.png")
        >>> print(scores['composition'])
        8.2
    """
    model, tokenizer = get_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Load and preprocess image
    pixel_values = load_image(image, max_num=12).to(dtype)
    if device == 'cuda':
        pixel_values = pixel_values.cuda()

    # Get expert scores
    expert_score, expert_text = model.expert_score(tokenizer, pixel_values)

    return expert_score, expert_text


def get_expert_annotations(image: Union[str, Path, Image.Image]) -> Dict[str, str]:
    """
    Get detailed expert annotations across 12 aesthetic dimensions.

    Args:
        image: Path to image file or PIL Image

    Returns:
        Dict with expert annotations for 12 dimensions

    Example:
        >>> annotations = get_expert_annotations("design.png")
        >>> print(annotations['composition'])
        "The composition is well-balanced with..."
    """
    model, tokenizer = get_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Load and preprocess image
    pixel_values = load_image(image, max_num=12).to(dtype)
    if device == 'cuda':
        pixel_values = pixel_values.cuda()

    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # Get expert annotations
    expert_annotation = model.expert_annotataion(tokenizer, pixel_values, generation_config)

    return expert_annotation
