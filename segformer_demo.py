"""
Reference: https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation.forward.example
"""
import os
import torch

from datasets import load_dataset
from transformers import AutoImageProcessor, SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_CHECKPOINT = "nvidia/mit-b0"

dataset = load_dataset("eborghi10/VineyardRows", trust_remote_code=True, token=os.getenv("HF_TOKEN"))
image = dataset["validation"]["pixel_values"][0]

image_size = image.size[::-1]
print("Image size: {image_size}")

image_processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_CHECKPOINT)

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

print(f"Outputs: {len(outputs)}")
print(f"Logits shape: {list(logits.shape)}")
