FROM docker.m.daocloud.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 1. 换成阿里云源 (通常比清华源在大文件下载上更稳)
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

# 2. 第一组：基础工具 (加入自动重试逻辑)
RUN apt-get update && \
    for i in {1..3}; do \
    apt-get install -y --fix-missing \
    wget htop git nano cmake unzip zip vim ninja-build \
    libegl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    libglew-dev libglfw3-dev libgtk-3-dev \
    libeigen3-dev libgl1-mesa-dev xorg-dev && break || \
    (echo "Retrying apt-get install... $i" && sleep 5 && apt-get update); \
    done && \
    apt-get clean

# 3. 第二组：重型库 (缓存层 2)
RUN apt-get update && (apt-get install -y --fix-missing \
    libopencv-dev \
    libassimp-dev assimp-utils \
    libboost-all-dev \
    ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev \
    libembree-dev || \
    (sleep 5 && apt-get install -y --fix-missing)) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN ln -s /lib/x86_64-linux-gnu/libembree3.so /lib/x86_64-linux-gnu/libembree.so
    
# 4. 修复 ENV 语法
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYOPENGL_PLATFORM=egl

# 5. fixuid 设置
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USERNAME\ngroup: $USERNAME\n" > /etc/fixuid/config.yml

USER $USERNAME:$USERNAME
WORKDIR /home/$USERNAME

# 6. Miniforge 安装
ENV PATH="/home/${USERNAME}/miniforge/bin:${PATH}"
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh -O ~/miniforge.sh && \
    chmod +x ~/miniforge.sh && \
    ~/miniforge.sh -b -p ~/miniforge && \
    rm ~/miniforge.sh

# 7. Conda 环境创建 (针对网络中断进行加固)
COPY docker/environment.yml /home/$USERNAME/environment.yml

# 增加超时时间到 10 分钟，并增加重试次数
RUN conda config --set remote_read_timeout_secs 600 && \
    conda config --set ssl_verify no && \
    conda config --remove-key channels || true && \
    conda config --add channels defaults && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia

# 使用循环重试创建环境。Conda 有断点续传功能，失败重试会继续下载剩下的包
RUN for i in {1..5}; do \
    conda env create -f /home/$USERNAME/environment.yml && break || \
    (echo "Conda installation failed, retrying in 10s... (Attempt $i)" && sleep 10); \
    done

# 8. 路径设置
ENV PATH="/home/${USERNAME}/miniforge/envs/bbsplat/bin:${PATH}"
RUN echo "source activate bbsplat" > ~/.bashrc

RUN /bin/bash --login -c "conda activate bbsplat && \
    pip install --no-cache-dir \
    mediapy==1.1.2 \
    opencv-python==4.9.0.80 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    trimesh \
    xatlas"

# 步骤 10: 安装 nvdiffrast (需要编译，加 --no-build-isolation)
RUN /bin/bash --login -c "conda activate bbsplat && \
    export TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9+PTX' && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"

# 步骤 11: 安装 PyTorch3D
# 同样需要指定架构，否则这一步也会报同样的错误
RUN /bin/bash --login -c "conda activate bbsplat && \
    export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;8.9+PTX' && \
    export MAX_JOBS=2 && \
    pip install git+https://github.com/facebookresearch/pytorch3d.git"
# Docker 启动配置
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]