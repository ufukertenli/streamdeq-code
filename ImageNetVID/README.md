# ImageNet-VID Experiments

This part of the repository handles video object detection experiments. 

## Installation

The code presented here merges [MMDetection](https://github.com/open-mmlab/mmdetection), [MMTracking](https://github.com/open-mmlab/mmtracking), and the [official implementation of MDEQ](https://github.com/locuslab/deq). In addition to MMDet, you will need to install the dependencies of the MDEQ repo. However, you do not need MMTracking for the code to work. 

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n streamdeq python=3.8 -y
    conda activate streamdeq
    ```

2. Install PyTorch (1.10.0) and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

3. Install mmcv-full and mmdetection.

    ```shell
    pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
    pip install mmdet==2.10.0
    ```

4. Install StreamDEQ.

	```shell
    python setup.py develop
    ```

## Usage

* Download ImageNET-VID & DET datasets from the [official source](http://image-net.org/challenges/LSVRC/2015/2015-downloads) and structure the paths as follows:

    ```
    ./data/ILSVRC/
    ./data/ILSVRC/Annotations/DET
    ./data/ILSVRC/Annotations/VID
    ./data/ILSVRC/Data/DET
    ./data/ILSVRC/Data/VID
    ./data/ILSVRC/ImageSets
    ```

**Note:** You may find the image sets [here](https://github.com/Scalsol/mega.pytorch/tree/master/datasets/ILSVRC2015/ImageSets).

### Training (4 GPUs Recommended)

1. Download the [ImageNet pretrained MDEQ model](https://drive.google.com/file/d/1vcWnlb5yUXE-3VHiuVJdfRZVeyx0U9-W/view) from the original MDEQ repository.

2. To perform multi-gpu training, you may run the following command:

    ```
    bash ./tools/dist_train.sh \
        configs/streamdeq/faster_rcnn_mdeq_fpn_1x_imagenetvid.py \
        ${GPU_NUM}
    ```

For more detailed information (training on single GPU, training with slurm) please visit the [documentation page of MMDetection](https://mmdetection.readthedocs.io/en/v2.10.0/1_exist_data_model.html).


### Evaluation

1. Download our [object detection model](https://drive.google.com/file/d/1mMcLgBZR9va3vYxG-E6ACKag49PrfXS3/view).

2. You may run the following commands to evaluate the pretrained model on streaming videos:
	
    ```
    python -u tools/test.py \
        configs/streamdeq/faster_rcnn_mdeq_fpn_1x_imagenetvid_stream_[NUM_FRAMES]f_[NUM_ITERS]i.py \
        path/to/pretrained/model.pth \
        --eval bbox
    ```

    where [NUM_FRAMES] is the length of each video and [NUM_ITERS] is the number of iterations performed per frame.
