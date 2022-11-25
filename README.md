# Streaming Multiscale Deep Equilibrium Models

This is the official implementation of the paper [Streaming Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2204.13492). Please visit our [project page](https://ufukertenli.github.io/streamdeq/) for more detailed information and qualitative results.

## Abstract

We present StreamDEQ, a method that infers frame-wise representations on videos with minimal per-frame computation. In contrast to conventional methods where compute time grows at least linearly with the network depth, we aim to update the representations in a continuous manner. For this purpose, we leverage the recently emerging implicit layer model which infers the representation of an image by solving a fixed-point problem. Our main insight is to leverage the slowly changing nature of videos and use the previous frame representation as an initial condition on each frame. This scheme effectively recycles the recent inference computations and greatly reduces the needed processing time. Through extensive experimental analysis, we show that StreamDEQ is able to recover near-optimal representations in a few frames time, and maintain an up-to-date representation throughout the video duration. Our experiments on video semantic segmentation and video object detection show that StreamDEQ achieves on par accuracy with the baseline (standard MDEQ) while being more than 3x faster.

## Qualitative Results

![Baseline with 2 iterations per frame](resources/Baseline_2_iteration.gif "" =50%x) *Baseline with 2 iterations per frame* ![StreamDEQ with 2 iterations per frame](resources/StreamDEQ_2_iteration.gif "" =50%x) *StreamDEQ with 2 iterations per frame*

## Citation

If you find this repository useful, please consider citing our work:

```
@InProceedings{ertenli2022streaming,
  author="Ertenli, Can Ufuk and Akbas, Emre and Cinbis, Ramazan Gokberk",
  title="Streaming Multiscale Deep Equilibrium Models",
  booktitle="European Conference on Computer Vision (ECCV)",
  year="2022",
  pages="189--205",
  organization={Springer}
}
```
