import torch
import torch.nn as nn

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class MDEQ_FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MDEQ_FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.training_iter = 0
        self.device_count = torch.cuda.device_count()

    def extract_feat(self, img, train_step=-1, compute_jac_loss=False, spectral_radius_mode=False, **kwargs):
        """Directly extract features from the backbone."""
        x = self.backbone(img, train_step, compute_jac_loss, spectral_radius_mode, **kwargs)
        return x

    def extract_feats(self, imgs):
        raise NotImplementedError

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x, _, _ = self.extract_feat(img)
        if self.with_neck:
            x = self.neck(x)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

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
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box.

            ref_img (Tensor): of shape (N, R, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                R denotes the number of reference image for each input
                image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:VideoCollect`.

            ref_gt_bboxes (list[Tensor]): The list only has one Tensor. The
                Tensor contains ground truth bboxes for each reference image
                with shape (num_all_ref_gts, 5) in
                [ref_img_id, tl_x, tl_y, br_x, br_y] format. The ref_img_id
                start from 0, and denotes the id of reference image for each
                key image.

            ref_gt_labels (list[Tensor]): The list only has one Tensor. The
                Tensor contains class indices corresponding to each reference
                box with shape (num_all_ref_gts, 2) in
                [ref_img_id, class_indices].

            gt_instance_ids (None | list[Tensor]): specify the instance id for
                each ground truth bbox.

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals (None | Tensor) : override rpn proposals with custom
                proposals. Use when `with_rpn` is False.

            ref_gt_instance_ids (None | list[Tensor]): specify the instance id
                for each ground truth bboxes of reference images.

            ref_gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes of reference images can be ignored when computing the
                loss.

            ref_gt_masks (None | Tensor) : True segmentation masks for each
                box of reference image used if the architecture supports a
                segmentation task.

            ref_proposals (None | Tensor) : override rpn proposals with custom
                proposals of reference images. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(img) == 1, \
            'Video detectors only support 1 batch size per gpu.'

        # compute jacobian loss weight (which is dynamically scheduled)
        deq_mode = True
        deq_steps = self.training_iter - self.backbone.pretrain_steps
        if deq_steps < 0:
            # We can also regularize output Jacobian when pretraining
            factor = self.backbone.pretrain_jac_loss_weight
            deq_mode = False
        elif self.training_iter >= self.backbone.jac_stop_iter:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            # Dynamically schedule the Jacobian reguarlization loss weight, if needed
            factor = self.backbone.jac_loss_weight + 0.1 * (deq_steps // self.backbone.update_freq)
        compute_jac_loss = (torch.rand([]).item() < self.backbone.jac_loss_freq) and (factor > 0)
        delta_f_thres = torch.randint(-self.backbone.rand_f_thres_delta, self.backbone.rand_f_thres_delta, []).item() if (
                self.backbone.rand_f_thres_delta > 0) else 0  # TODO: Removed compute_jac_loss condition from here
        f_thres = self.backbone.f_thres + delta_f_thres
        b_thres = self.backbone.b_thres

        x, jac_losses, _ = self.extract_feat(img, train_step=self.training_iter, compute_jac_loss=compute_jac_loss,
                                             f_thres=f_thres, b_thres=b_thres, deq_mode=deq_mode)
        jac_loss = jac_losses.mean()

        if self.with_neck:
            x = self.neck(x)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        losses.update({'jac_loss': factor * jac_loss})

        self.training_iter += 1 * self.device_count

        return losses

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

        x, _, _ = self.extract_feat(img)

        if self.with_neck:
            x = self.neck(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

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
