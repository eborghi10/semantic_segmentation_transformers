"""
Reference: https://huggingface.co/docs/transformers/main/en/tasks/semantic_segmentation#custom-dataset
"""

from datasets import Dataset, DatasetDict, Image
from glob import glob
import os

image_paths_train = glob("datasets/VineyardRows/train/images/*.jpg")
label_paths_train = glob("datasets/VineyardRows/train/labels/*.jpg")

image_paths_validation = glob("datasets/VineyardRows/validation/images/*.jpg")
label_paths_validation = glob("datasets/VineyardRows/validation/labels/*.jpg")

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict({
     "train": train_dataset,
     "validation": validation_dataset,
     }
)

# step 3: push to Hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
dataset.push_to_hub("eborghi10/VineyardRows", private=True, token=os.getenv("HF_TOKEN"))
