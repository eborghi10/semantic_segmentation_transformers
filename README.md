# Semantic Segmentation with Vision Transformers (ViT)

This repository uses the **SegFormer** model proposed in _[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo_.

## Prerequisites

Create a `.env` file in the root directory of this repository with your Hugging Face token:

```env
HF_TOKEN=<your_token>
CUDA_LAUNCH_BLOCKING=1
```

## Usage

Open this repository as a [VS Devcontainer](https://code.visualstudio.com/docs/devcontainers/tutorial) and open the following demos:

* Dataset: `dataset.py`. Will push the dataset to Hugging Face.
* Training: `segformer-vineyard-train.ipynb`.
* Inference: TODO, see below.

## Known issues

* Training fails with custom dataset.

## To do

* Use Docker image in Devcontainer
* Create inference on custom dataset
* Setup MLOps pipeline

## References

* [Semantic segmentation examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation)
