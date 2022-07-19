import torch

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class StreamDEQ(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 extra=None):
        super(StreamDEQ, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.num_frames = extra['num_frames']
        self.f_thres = extra['f_thres']

        self.training_iter = 0

    def extract_feat(self, img, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, **kwargs):
        """Directly extract features from the backbone."""
        x = self.backbone(img, train_step, compute_jac_loss, spectral_radius_mode, **kwargs)
        return x

    def extract_feats(self, imgs):
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        raise NotImplementedError  # Video training is not supported

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (list[Tensor] | None): The list only contains one Tensor
                of shape (1, N, C, H, W) encoding input reference images.
                Typically these should be mean centered and std scaled. N
                denotes the number for reference images. There may be no
                reference images in some cases.

            ref_img_metas (list[list[list[dict]]] | None): The first and
                second list only has one element. The third list contains
                image information dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain 'filename',
                'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
                the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

            proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

            rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_refs = ref_img[0].shape[1] if ref_img is not None else None

        if num_refs is None:
            num_frames = None
        else:
            # limit the number of frames to the number of reference frames
            num_frames = num_refs if self.num_frames > num_refs else self.num_frames

        if ref_img is None:  # Single image processing
            x, _, _ = self.extract_feat(img, f_thres=self.f_thres)
        else:  # Streaming video processing
            inp_list = []  # keeps track of the previous frame's representation
            for i in range(num_frames):
                if not inp_list:  
                # For the first frame of a video, we start from scratch
                    ref_x, _, _ = self.extract_feat(ref_img[0][:, num_refs - num_frames + i], f_thres=self.f_thres)
                    if num_frames == 1:
                        break
                else:  
                # For frames after the first frame, we use the previous representation as the starting point
                    ref_x, _, _ = self.extract_feat(inp_list, f_thres=self.f_thres)
                    if (num_refs - num_frames + i + 1) >= num_refs:
                        break
                inp_list = [ref_img[0][:, num_refs - num_frames + i + 1]]
                inp_list.extend(list(ref_x))
            inp_list = [img]
            inp_list.extend(list(ref_x))
            # Last frame of the video is processed to make predictions
            x, _, _ = self.extract_feat(inp_list, f_thres=self.f_thres)

        if self.with_neck:
            x = self.neck(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        raise NotImplementedError
