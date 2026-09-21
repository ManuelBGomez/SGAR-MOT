# SGAR-MOT -- container image.
#
# Reproduces the environment the released code was verified in.
# The build context must be the root of this repository. See TUTORIAL.md for more information.

FROM nvcr.io/nvidia/tensorrt:23.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG WORKDIR=/workspace/ByteTrack
ARG BYTETRACK_COMMIT=3434c5e8bc6a5ae8ad530528ba8d9a431967f237
ARG SAM2_COMMIT=7e1596c0b6462eb1d1ba7e1492430fed95023598

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl unzip pkg-config \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch first, so nothing downgrades it later.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1

# Versions of the verified environment, with one exception: PyYAML is 6.0.2 here
# rather than the 5.4.1 of that environment, because 5.4.1 no longer builds against
# Cython 3.
RUN pip3 install --no-cache-dir \
        numpy==1.23.5 \
        opencv-python==4.10.0.84 \
        Pillow==10.1.0 \
        scipy==1.14.1 \
        matplotlib==3.9.2 \
        pandas==2.2.2 \
        PyYAML==6.0.2 \
        loguru==0.7.2 \
        tqdm==4.66.5 \
        tabulate==0.9.0 \
        tensorboard==2.17.1 \
        scikit-learn==1.6.0 \
        scikit-image==0.24.0 \
        torchmetrics==1.6.0 \
        motmetrics==1.4.0 \
        filterpy==1.4.5 \
        h5py==3.11.0 \
        lap==0.4.0 \
        thop==0.1.1.post2209072238 \
        ninja==1.11.1.1 \
        hydra-core==1.3.2 \
        iopath==0.1.10 \
    && pip3 install --no-cache-dir Cython==3.0.11 \
    && pip3 install --no-cache-dir cython-bbox==0.1.5 \
        'git+https://github.com/cocodataset/cocoapi.git@8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9#subdirectory=PythonAPI'

# ByteTrack + YOLOX (builds the yolox CUDA extension).
RUN git clone https://github.com/ifzhang/ByteTrack.git ${WORKDIR} \
    && cd ${WORKDIR} \
    && git checkout ${BYTETRACK_COMMIT} \
    && python3 setup.py develop

# SAM 2, pinned to the release SGAR-MOT was developed against:
RUN git clone https://github.com/facebookresearch/sam2.git ${WORKDIR}/sam2 \
    && cd ${WORKDIR}/sam2 \
    && git checkout ${SAM2_COMMIT}
COPY src/sam2_sgar_mot/sam2/ ${WORKDIR}/sam2/sam2/

# Fail loudly if the CUDA extension does not build: without `sam2._C`, SAM 2
# silently skips mask post-processing and the masks change.
ENV SAM2_BUILD_ALLOW_ERRORS=0

ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN cd ${WORKDIR}/sam2 \
    && pip3 install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python3 -c "import torch; from sam2 import _C; print('sam2._C OK')"

# SGAR-MOT itself, plus the tracker files that go into the ByteTrack checkout.
COPY . /opt/sgar-mot
RUN pip3 install --no-cache-dir --no-deps -e /opt/sgar-mot \
    && cp -r /opt/sgar-mot/src/bytetrack/. ${WORKDIR}/ \
    && cp -r /opt/sgar-mot/src/configs ${WORKDIR}/configs

WORKDIR ${WORKDIR}
CMD ["/bin/bash"]
