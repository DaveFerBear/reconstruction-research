"""UIClip model for predicting UI aesthetic quality scores."""

import os
# Disable tokenizers parallelism to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force PyTorch backend (disable TensorFlow)
os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)  # Limit to single thread to avoid issues

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from typing import Union

IMG_SIZE = 224
DEVICE = "cpu"  # can also be "cuda" or "mps"
LOGIT_SCALE = 100  # based on OpenAI's CLIP example code
NORMALIZE_SCORING = True

MODEL_PATH = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
PROCESSOR_PATH = "openai/clip-vit-base-patch32"

# Lazy load model and processor
_model = None
_processor = None


def get_model():
    """Lazy load the UIClip model."""
    global _model, _processor
    if _model is None:
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        _model = CLIPModel.from_pretrained(MODEL_PATH)
        _model = _model.eval()
        _model = _model.to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(PROCESSOR_PATH)
    return _model, _processor


def preresize_image(image, image_size):
    """Resize image maintaining aspect ratio."""
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        image = image.resize((int(aspect_ratio * image_size), image_size))
    else:
        image = image.resize((image_size, int(image_size / aspect_ratio)))
    return image


def slide_window_over_image(input_image, img_size):
    """Apply sliding window to handle different image sizes."""
    input_image = preresize_image(input_image, img_size)
    width, height = input_image.size
    square_size = min(width, height)
    longer_dimension = max(width, height)
    num_steps = (longer_dimension + square_size - 1) // square_size

    if num_steps > 1:
        step_size = (longer_dimension - square_size) // (num_steps - 1)
    else:
        step_size = square_size

    cropped_images = []

    for y in range(0, height - square_size + 1, step_size if height > width else square_size):
        for x in range(0, width - square_size + 1, step_size if width > height else square_size):
            left = x
            upper = y
            right = x + square_size
            lower = y + square_size
            cropped_image = input_image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)

    return cropped_images


def compute_description_embeddings(descriptions):
    """Compute text embeddings for descriptions."""
    model, processor = get_model()
    inputs = processor(text=descriptions, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
    inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)
    text_embedding = model.get_text_features(**inputs)
    return text_embedding


def compute_image_embeddings(image_list):
    """Compute image embeddings using sliding window approach."""
    model, processor = get_model()
    windowed_batch = [slide_window_over_image(img, IMG_SIZE) for img in image_list]
    inds = []
    for imgi in range(len(windowed_batch)):
        inds.append([imgi for _ in windowed_batch[imgi]])

    processed_batch = [item for sublist in windowed_batch for item in sublist]
    inputs = processor(images=processed_batch, return_tensors="pt")
    # run all sub windows of all images in batch through the model
    inputs['pixel_values'] = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # output contains all subwindows, need to mask for each image
    processed_batch_inds = torch.tensor([item for sublist in inds for item in sublist]).long().to(image_features.device)
    embed_list = []
    for i in range(len(image_list)):
        mask = processed_batch_inds == i
        embed_list.append(image_features[mask].mean(dim=0))
    image_embedding = torch.stack(embed_list, dim=0)
    return image_embedding


def compute_quality_scores(input_list):
    """
    Compute quality scores for a list of (description, image) pairs.

    Args:
        input_list: List of tuples (description: str, image: PIL.Image)

    Returns:
        torch.Tensor: Quality scores for each input (0-1 if normalized)
    """
    # input_list is a list of types where the first element is a description and the second is a PIL image
    description_list = ["ui screenshot. well-designed. " + input_item[0] for input_item in input_list]
    img_list = [input_item[1] for input_item in input_list]
    text_embeddings_tensor = compute_description_embeddings(description_list)  # B x H
    img_embeddings_tensor = compute_image_embeddings(img_list)  # B x H

    # normalize tensors
    text_embeddings_tensor /= text_embeddings_tensor.norm(dim=-1, keepdim=True)
    img_embeddings_tensor /= img_embeddings_tensor.norm(dim=-1, keepdim=True)

    if NORMALIZE_SCORING:
        text_embeddings_tensor_poor = compute_description_embeddings([d.replace("well-designed. ", "poor design. ") for d in description_list])  # B x H
        text_embeddings_tensor_poor /= text_embeddings_tensor_poor.norm(dim=-1, keepdim=True)
        text_embeddings_tensor_all = torch.stack((text_embeddings_tensor, text_embeddings_tensor_poor), dim=1)  # B x 2 x H
    else:
        text_embeddings_tensor_all = text_embeddings_tensor.unsqueeze(1)

    img_embeddings_tensor = img_embeddings_tensor.unsqueeze(1)  # B x 1 x H

    scores = (LOGIT_SCALE * img_embeddings_tensor @ text_embeddings_tensor_all.permute(0, 2, 1)).squeeze(1)

    if NORMALIZE_SCORING:
        scores = scores.softmax(dim=-1)

    return scores[:, 0]


def score_image(image: Union[str, Path, Image.Image], description: str = "") -> float:
    """
    Compute quality score for a single image.

    Args:
        image: Path to image file or PIL Image
        description: Optional description of the design (e.g., "social media post", "landing page")

    Returns:
        float: Quality score (0-1 if normalized)

    Example:
        >>> score_image("design.png", "social media post")
        0.85
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    scores = compute_quality_scores([(description, image)])
    return scores[0].item()
