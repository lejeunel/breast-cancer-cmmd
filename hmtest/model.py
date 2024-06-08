import math
from collections import OrderedDict
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.models.detection import _utils as det_utils
from torchvision.models.resnet import ResNet18_Weights
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.boxes import box_iou

from obj_detection.anchor_generator import AnchorGenerator


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class BBoxClassificationHead(nn.Module):
    """
    A classification head to predict anchor box probabilities.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization
                        layer to use. Default: None
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_focal_loss: bool = True,
    ):
        super().__init__()
        self.use_focal_loss = use_focal_loss

        conv = []
        for _ in range(4):
            conv.append(
                misc_nn_ops.Conv2dNormActivation(
                    in_channels, in_channels, norm_layer=norm_layer
                )
            )
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(
            self.cls_logits.bias,
            -math.log((1 - prior_probability) / prior_probability),
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, cls_logits, matched_idxs):
        losses = []

        for cls_logits_per_image, matched_idxs_per_image in zip(
            cls_logits, matched_idxs
        ):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification (one-hot encodings)
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[foreground_idxs_per_image, 1] = 1.0
            gt_classes_target[~foreground_idxs_per_image, 0] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = (
                matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            )

            valid_cls_logits = cls_logits_per_image[valid_idxs_per_image]
            valid_targets = gt_classes_target[valid_idxs_per_image]
            if self.use_focal_loss:
                # compute the classification loss
                losses.append(
                    sigmoid_focal_loss(
                        valid_cls_logits,
                        valid_targets,
                        reduction="sum",
                    )
                    / max(1, num_foreground)
                )
            else:
                losses.append(
                    binary_cross_entropy_with_logits(
                        valid_cls_logits, valid_targets, reduction="sum"
                    )
                    / max(1, num_foreground)
                )

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(
                N, -1, self.num_classes
            )  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class CosineModule(nn.Module):
    """
    Transform feature vector into an embedding for cosine softmax classification

    Args:
        in_dim (int): number of channels of the input feature
        out_dim (int): dimension of embeddings
        n_classes (int): number of clusters to learn
    """

    def __init__(self, in_dim, units, out_dim, n_classes, cosine_relu=False):
        super().__init__()
        self.units = units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_classes = n_classes

        self.cosine_relu = cosine_relu

        self.proj = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, units, bias=False),
            nn.ReLU(),
            nn.Linear(units, units, bias=False),
            nn.ReLU(),
            nn.Linear(units, out_dim, bias=False),
        )

        self.clusters = nn.Linear(self.out_dim, self.n_classes, bias=False)
        self.clusters_w = nn.Parameter(self.clusters.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(self.in_dim, 1)

    def forward(self, input):
        """
        Returns:
            - cosine (Tensor): Cosines between embedding and each cluster centroid
            - x (Tensor): Embedding
        """
        x = F.normalize(self.proj(input))
        w = F.normalize(self.clusters_w).T

        if self.cosine_relu:
            x = F.relu(x)
            w = F.relu(w)
        scale = torch.exp(self.bn_scale(self.fc_scale(input)))
        cosine = scale * torch.mm(x, w)
        return cosine, x


class MyModel(nn.Module):
    """

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects input tensors target bounding boxes, and class-labels, where
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``IntTensor[B]``): the ground-truth classes.

    During inference, the model requires only the input tensors.

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        num_clusters (int): number of cluster to learn

    """

    __annotations__ = {
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        anchor_generator,
        num_clusters=100,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.4,
        cos_units_dims=512,
        embedding_dims=128,
        use_pretrained_weights=True,
        use_focal_loss=True,
        **kwargs,
    ):
        super().__init__()

        self.backbone = torchvision.models.resnet18(
            weights=ResNet18_Weights.DEFAULT
            if use_pretrained_weights
            else None
        )
        self.backbone.out_channels = 512

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or\
                None instead of {type(anchor_generator)}"
            )

        self.anchor_generator = anchor_generator

        self.head = BBoxClassificationHead(
            self.backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes=2,
            use_focal_loss=use_focal_loss,
        )

        self.cos_softmax = CosineModule(
            self.backbone.out_channels,
            cos_units_dims,
            embedding_dims,
            num_clusters,
        )

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        # apply rescaling to match pre-training dataset (imagenet)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = torchvision.transforms.Normalize(
            image_mean, image_std
        )

        # used only on torchscript mode
        self._has_warned = False

    def compute_bbox_loss(self, targets, head_outputs, anchors):
        """
        Generate targets for each anchor boxes candidates given ground-truth
        bounding boxes
        through a thresholding function
        """

        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image.numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue

            match_quality_matrix = box_iou(
                targets_per_image, anchors_per_image
            )
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, matched_idxs)

    def compute_cosine_softmax_loss(self, cosines, targets):
        """
        Returns cross-entropy loss on cosines using class labels given in targets tensor.
        """
        return F.cross_entropy(cosines, targets)

    def _forward_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def _get_most_likely_feats(
        self, image_shape, features, anchors, head_outputs
    ):
        # find center location of each anchor box
        feat_size = features.shape[2]
        decim_factor = image_shape // feat_size
        cx = (anchors[0][:, 2] - anchors[0][:, 0]) / decim_factor
        cy = (anchors[0][:, 3] - anchors[0][:, 1]) / decim_factor
        centers_ij = torch.stack((cy, cx))

        # find index of most likely anchor box on each image
        idx_most_likely = head_outputs.softmax(dim=-1)[..., -1].argmax(dim=-1)

        # get feature where object most likely lies
        local_feats = []
        for f, idx_box in zip(features, idx_most_likely):
            i, j = centers_ij[:, idx_box]
            i, j = torch.round(i).int(), torch.round(j).int()
            i, j = torch.clamp(i, min=0, max=feat_size - 1), torch.clamp(
                j, min=0, max=feat_size - 1
            )
            local_feats.append(f[:, i, j])

        return torch.stack(local_feats)

    def forward(self, images, targets_boxes=None, classes=None):
        """
        Args:
            images (Tensor): images to be processed
            targets_boxes (Tensor): coordinates of ground-truth boxes (optional)
            classes (Tensor): class id of each image for metric learning

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        # get the original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        if self.transform:
            images = self.transform(images)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets_boxes is not None:
            for target_idx, target in enumerate(targets_boxes):
                boxes = target
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self._forward_backbone(images)

        # we have a single pyramid level in this implementation
        features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features[0])

        local_feats = self._get_most_likely_feats(
            images.shape[2], features[0], anchors, head_outputs
        )
        cosines, embeddings = self.cos_softmax(local_feats)

        losses = {}
        if self.training:
            if targets_boxes is None:
                torch._assert(
                    False, "targets should not be none when in training mode"
                )
            else:
                bbox_loss = self.compute_bbox_loss(
                    targets_boxes, head_outputs, anchors
                )
                cos_loss = self.compute_cosine_softmax_loss(cosines, classes)
                losses["bbox_clf"] = bbox_loss
                losses["cosine_softmax"] = cos_loss

        return {
            "losses": losses,
            "anchors": anchors,
            "outputs": {
                "bbox_logits": head_outputs,
                "embeddings": embeddings,
                "features": local_feats,
                "cosines": cosines,
            },
        }


def make_model(
    num_clusters=100,
    anchor_sizes=((250, 300, 350, 400),),
    aspect_ratios=((0.5, 0.6, 0.7),),
    fg_iou_thresh=0.75,
    bg_iou_thresh=0.4,
    cos_unit_dims=512,
    cos_embedding_dims=128,
    pretrained=False,
    use_focal_loss=True,
):
    """
    Build object detection model
    """
    from obj_detection.anchor_generator import AnchorGenerator

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios
    )
    model = MyModel(
        num_clusters=num_clusters,
        anchor_generator=anchor_generator,
        fg_iou_thresh=fg_iou_thresh,
        bg_iou_thresh=bg_iou_thresh,
        cos_units_dims=cos_unit_dims,
        embedding_dims=cos_embedding_dims,
        use_pretrained_weights=pretrained,
        use_focal_loss=use_focal_loss,
    )
    return model
