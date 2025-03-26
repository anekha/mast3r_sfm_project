# MASt3R_SfM_Project

This project implements the **MASt3R** paper: *Grounding Image Matching in 3D with MASt3R*.

- **[Project Page](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)**
- **[MASt3R ArXiv](https://arxiv.org/abs/2406.09756)**
- **[DUSt3R ArXiv](https://arxiv.org/abs/2312.14132)**

## Overview

This project aims to estimate accurate camera parameters from multi-view images without requiring prior calibration. Traditional methods like COLMAP often fail when camera parameters are unknown. MASt3R provides a robust solution using deep learning techniques.

## Features

- Multi-view stereo reconstruction without known camera parameters.
- Transformer-based architecture for feature matching.
- Dense and unconstrained stereo 3D reconstruction.

## Installation

1. Clone this repository:
   ```bash
   git clone --recurse-submodules git@github.com:anekha/mast3r_sfm_project.git
   
2. Make sure you have the required dependencies installed. We recommend using a virtual environment (e.g., venv or conda).

## Dependencies

You will need to install the necessary Python packages and system dependencies.
1.	Install System Dependencies:
- Follow the system setup steps (if applicable) for CUDA, Python 3.11, and other required libraries.
- You may need to install additional system libraries like gcc-10, libomp-dev, ninja-build, and ffmpeg.
2. Install Python Dependencies:
- Install the Python packages using pip from requirements.txt:

```bash
pip install -r requirements.txt
```

## Submodules
- The repository uses submodules, which can be initialized with the following:
```bash
git submodule update --init --recursive
```
## 	Set up Hugging Face:
- You may need to set up Hugging Face integration for downloading pre-trained models. 
- Follow instructions on Hugging Face to create a token and add it to your environment.

## Running the Project

After setting up, you can start processing images and performing 3D reconstruction by running the modal_run.py script. Here’s a basic example of how to use the code:
1. Prepare your images (12-50 images recommended) in a folder. 
2. Run the following command:

```bash
python modal_run.py --input_folder <path_to_images> --output_folder <path_to_output>

```
You may need to adjust the parameters or add additional flags depending on the specifics of your dataset.

## Checkpoints

To use pre-trained models, download the following checkpoints:
 ```bash
 mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/```
```
## Contributing

Feel free to fork this project, report issues, and submit pull requests. 