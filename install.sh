#!/usr/bin/env bash
# PyTorch with CUDA 12.8 (choose the build for your GPU at https://pytorch.org/get-started/locally/)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
