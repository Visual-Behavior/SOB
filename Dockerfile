FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
RUN pip install kornia ipython scikit-image tensorboard tensorboardX jupyter einops seaborn accelerate timm torchmetrics rich numpy==2.1.2
