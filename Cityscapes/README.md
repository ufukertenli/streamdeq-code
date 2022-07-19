# Cityscapes Experiments

This part of the repository handles video semantic segmentation experiments. Most of the code here is from the [original DEQ repo](https://github.com/locuslab/deq). For installation instructions and detailed explanations about the usage of the code please visit the original DEQ repo. 

## Dataset

The Cityscapes dataset is usually used with single image training. However, the official website provides snippets from which the train, val, and test sets are created. The dataset split we use for this purpose is ``leftImg8bit_sequence_trainvaltest.zip``. You can download the dataset using [CityscapesScripts](https://github.com/mcordts/cityscapesScripts) repo. 

For video evaluation, we provide dataset lists in the ``data/list/`` folder.

## Usage

1. Download the pretrained MDEQ Segmentation model from [here](https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view).

2. Run the following command to test StreamDEQ on streaming videos:

    ```
    python tools/seg_test.py --cfg experiments/cityscapes/seg_MDEQ_XL_[NUM_FRAMES]f_[NUM_ITERS]i.yaml
    ```

    where [NUM_FRAMES] is the length of each video and [NUM_ITERS] is the number of iterations performed per frame. Note that, ``sf`` in the config names refer to the singleframe baseline cases. Those configs are equivalent to the config files in the original MDEQ.
