# Base Images
# docker build流程
# docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# docker build -t miccai2d .
# docker run --gpus=all miccai2d
# docker save -o miccai2d.tar miccai2d

# 解压miccai2d_docker.zip文件
# 在miccai2d_docker文件内打开git bash 或者cd <path/to/miccai2d_docker>
# mkdir outputs
# docker load --input miccai2d.tar
# docker run -it --name='miccai2d_container' --gpus=all miccai2d
# docker cp miccai2d_container:/infers_fusai/ outputs
## 从基础镜像构建
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
## 把当前文件夹里的文件构建到镜像的根目录下（.后面有空格，不能直接跟/）
ADD . /
## 指定默认工作目录为根目录（需要把infer.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
## Install Requirements
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

## 镜像启动后统一执行 sh infer.sh
CMD ["sh", "infer.sh"]

