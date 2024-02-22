# Semantic-Preserving Image Coding based on Conditional Diffusion Models (SPIC)

<img src='assets\scheme.png'>  

### [Paper](https://arxiv.org/abs/2310.15737)

[Francesco Pezone](https://scholar.google.com/citations?hl=it&user=RAOXtOEAAAAJ), [Osman Musa](https://scholar.google.com/citations?hl=it&user=a8y7ME8AAAAJ), [Giuseppe Caire](https://scholar.google.com/citations?hl=it&user=g66ErTcAAAAJ), [Sergio Barbarossa](https://scholar.google.com/citations?hl=it&user=2woHFu8AAAAJ)

## Abstract

Semantic communication, rather than on a bit-by-bit recovery of the transmitted messages, focuses on the meaning and the goal of the communication itself. In this paper, we propose a novel semantic image coding scheme that preserves the semantic content of an image, while ensuring a good trade-off between coding rate and image quality. The proposed Semantic-Preserving Image Coding based on Conditional Diffusion Models (SPIC) transmitter encodes a Semantic Segmentation Map (SSM) and a low-resolution version of the image to be transmitted. The receiver then reconstructs a high-resolution image using a Denoising Diffusion Probabilistic Models (DDPM) doubly conditioned to the SSM and the low-resolution image. As shown by the numerical examples, compared to state-of-the-art (SOTA) approaches, the proposed SPIC exhibits a better balance between the conventional rate-distortion trade-off and the preservation of semantically-relevant features.



## Prerequisites
- Operating System: Linux
- Python Version: Python 3
- Hardware: CPU or NVIDIA GPU with CUDA CuDNN

## Dataset Preparation
To utilize the Cityscapes dataset:

1. Follow the instructions provided by [SPADE](https://github.com/NVlabs/SPADE.git) for downloading and preparing the dataset.
2. Optionally, create a coarse version of the original images. By default, when applying the Semantic-Conditioned Super-Resolution Diffusion Model, the coarse images are generated using the [BPG](https://bellard.org/bpg/) compression algorithm at compression level 35.

## Training and Test

### Training

Download the dataset and initiate training with the following command:
```bash
python3 image_train.py --data_dir ./data/cityscapes  --dataset_mode cityscapes --lr 1e-4 --batch_size 8 --attention_resolutions 32,16,8 --diffusion_steps 1000  --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 19  --class_cond True --large_size 128 --small_size 64 --no_instance True
```
### Fine-tuning
To fine-tune the model:
```bash
python3 image_train.py --data_dir ./data/cityscapes  --dataset_mode cityscapes --lr 1e-4 --batch_size 8 --attention_resolutions 32,16,8 --diffusion_steps 1000  --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 19 --class_cond True --large_size 128 --small_size 64 --no_instance True --resume_checkpoint ./Checkpoints/model.pt
```

### Testing
To test the model:
```bash
python3 image_sample.py --data_dir ./data/cityscapes --dataset_mode cityscapes --batch_size 2 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 19 --class_cond True --large_size 128 --small_size 64 --no_instance True --num_samples 60 --s 1.5 --max_iter_time 200 --timestep_respacing 100 --no_instance True --model_path ./Checkpoints/model.pt --results_path ./Result/
```

## Pre-trained Model
Access the pretrained model for the Cityscapes dataset via the link provided below:

|Dataset       |Download link     |
|:-------------|:-----------------|
|Cityscapes|[Checkpoint](https://drive.google.com/file/d/1OTVp4KtGT75PRwWHmbtFRlTzpHjJwmPQ/view?usp=sharing)|


## Evaluation

To evaluate the performance of the model, follow these steps to calculate the Bits Per Pixel (BPP), mean Intersection over Union (mIoU), and the Fréchet Inception Distance (FID):

### Pre-requisites for Evaluation

Ensure you have the following components installed and set up:

1. **Semantic Segmentation Algorithm**: For generating Semantic Segmentation Maps (SSM). In our case, we utilized [InternImage](https://github.com/OpenGVLab/InternImage/tree/master).
2. **FLIF Compression Tool**: [FLIF](https://flif.info/), used for the lossless compression of the SSMs
3. **FID Calculation Tool**: Install [pytorch_fid](https://pypi.org/project/pytorch-fid/) to compute the Fréchet Inception Distance

### Steps for Evaluation

1. **Generate Test Results**: Before proceeding with the evaluation metrics, generate the test results using the model. Replace `./Checkpoints/model.pt` with the path to your model checkpoint and `./Result/` with the path where you want to save the results:
```bash
python3 image_sample.py --data_dir ./data/cityscapes --dataset_mode cityscapes --batch_size 2 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 19 --class_cond True --large_size 128 --small_size 64 --no_instance True --num_samples 60 --s 1.5 --max_iter_time 200 --timestep_respacing 100 --no_instance True --model_path ./Checkpoints/model.pt --results_path ./Result/
```

2. **Configure Evaluation Script**: In the `evaluate/metrics.py` file, update the paths to point to your results and the InternImage folder. Specifically, replace `path/to/Result` and `path/to/InternImage` with the actual paths on your system.

3. **Run Evaluation Metrics**: Execute the following command to calculate the BPP, mIoU, and FID for your generated images:
```bash
python evaluate/metrics.py
```


### Acknowledge
Our code is developed based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [Semantic Image Synthesis via Diffusion Models](https://github.com/WeilunWang/semantic-diffusion-model/tree/main). 
