import subprocess
import sys
import importlib
from PIL import Image
import numpy as np
import io
import modal

# Reference your volume
mast3r_projectvolume = modal.Volume.from_name("mast3rsfmvolume")

# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
huggingface_secret = Secret.from_name("my-huggingface-secret")

# ::::::: Image :::::::
cuda_version = "12.1.1"  # Adjust to the correct version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

mast3r_image = (
    modal.Image.from_registry(
    f"nvidia/cuda:{tag}",
    add_python="3.11"
    )
    .run_commands(
        [
            "set -e",  # Exit on any command failure

            # Install system dependencies
            "apt-get update && apt-get install -y "
            "build-essential libomp-dev libgl1 libglvnd0 libglib2.0-0 libsm6 "
            "libxext6 libxrender1 libjpeg-dev libpng-dev libtiff-dev ffmpeg "
            "libglm-dev clang git wget libboost-all-dev libboost-program-options-dev "
            "libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev "
            "libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev "
            "libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev "
            "libcgal-dev libceres-dev ninja-build gcc-10 g++-10 && "
            "apt-get clean && rm -rf /var/lib/apt/lists/*",

            # Set GCC-10 as the default compiler
            "export CC=/usr/bin/gcc-10",
            "export CXX=/usr/bin/g++-10",
            "export CUDAHOSTCXX=/usr/bin/g++-10",

            # Install CMake 3.28.0
            "wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh",
            "chmod +x cmake-3.28.0-linux-x86_64.sh",
            "./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local",
            "rm cmake-3.28.0-linux-x86_64.sh",
            "cmake --version",  # Verify CMake installation

            # Install Python dependencies
            "pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121",
            "pip install torch torchvision roma gradio matplotlib tqdm opencv-python scipy einops trimesh tensorboard 'pyglet<2'",
            "pip install huggingface-hub[torch]>=0.22 pillow-heif pyrender kapture kapture-localization numpy-quaternion pycolmap poselib scikit-learn cython",

            # Compile and install ASMK
            "git clone https://github.com/jenicek/asmk",
            "cd asmk/cython && cythonize *.pyx",  # Compile `.pyx` files
            "cd asmk && python3 setup.py build_ext --inplace",  # Build extensions in-place

            # Clone the GloMAP repository
            "git clone --recursive https://github.com/colmap/glomap.git",
            "cd glomap && mkdir -p build",
            "cd glomap/build && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native",
            "cd glomap/build && ninja && ninja install",

            # Clone DUST3R repository with submodules
            "git clone --recursive https://github.com/naver/dust3r.git",
            "cd dust3r && git submodule update --init --recursive",
            # Compile RoPE positional embeddings
            "cd dust3r/croco/models/curope && TORCH_CUDA_ARCH_LIST='8.0 8.6' python setup.py build_ext --inplace",

            # Verify CUDA toolkit installation
            "echo 'Checking CUDA toolkit installation...'",
            "nvcc --version",
            "which nvcc",

            # Verify Python 3.11 installation
            "echo 'Checking Python version...'",
            "python3.11 --version || echo 'Python 3.11 is not installed'",

            # Check PyTorch availability and version
            "echo 'Checking PyTorch installation...'",
            "python3.11 -c 'import torch; print(\"PyTorch version:\", torch.__version__)' || echo 'PyTorch is not installed or unavailable in Python 3.11'",

            # Verify installed Python packages
            "python3.11 -m pip list",

            # Final check message
            "echo 'Environment setup complete.'"
        ]
    )
)

# ::::::: Modal App :::::::
app = modal.App("mast3r_app")

# ::::::: Main Function :::::::
@app.function(volumes={"/my_vol_mast3r": mast3r_projectvolume},
              gpu="A10G",
              timeout=86400,
              secrets=[huggingface_secret],
              image=mast3r_image)

def main_function(
    image_folder="/my_vol_mast3r/mast3r_sfm/data/ring_one_40frames/images",
    model_checkpoint="/my_vol_mast3r/mast3r_sfm/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    output_dir="/my_vol_mast3r/output",
    image_size=(512, 512),
    device="cuda"
):
    """
    Main function to run MASt3R with Modal.
    """
    import subprocess

    # Construct the command to run the script
    command = [
        "python3.11",
        "/my_vol_mast3r/mast3r_sfm/run_mast3r.py",
        "--image_folder", image_folder,
        "--model_checkpoint", model_checkpoint,
        "--output_dir", output_dir,
        "--image_size", f"{image_size[0]},{image_size[1]}" if isinstance(image_size, tuple) else str(image_size),
        "--device", device,
    ]

    # Run the command
    subprocess.run(command, check=True)