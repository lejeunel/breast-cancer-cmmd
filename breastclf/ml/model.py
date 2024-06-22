from torch import nn
import torch
import torchvision
from breastclf.ml.shared import Batch


def make_resnet34_feat_extractor() -> tuple[nn.Module, int]:
    """
    Replace input layer with a single-channel conv2d,
    remove the last classification layer,
    and extract the dimensions of the feature vector
    """
    rn = torchvision.models.resnet34()
    feat_extractor = nn.Sequential(
        nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
            dilation=2,
            bias=False,
        ),
        rn.bn1,
        rn.relu,
        nn.MaxPool2d(kernel_size=3, stride=4, padding=1, dilation=1, ceil_mode=False),
        rn.layer1,
        rn.layer2,
        rn.layer3,
        rn.layer4,
        rn.avgpool,
    )
    n_features = feat_extractor[-2][-1].conv2.out_channels

    return feat_extractor, n_features


ABNORMALITY_BRANCH_NAMES = ["calcification", "mass"]


class BreastClassifier(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.0,
        fusion_mode="features",
        n_hidden_units=128,
        loss_factors={"type": 1, "abnorm": 1},
        type_weights={"Benign": 1 / 0.3, "Malignant": 1 / 0.7},
    ):
        super().__init__()

        allowed_fusion_modes = ["mean-feats", "max-feats", "concat-feats", "output"]
        assert (
            fusion_mode in allowed_fusion_modes
        ), f"fusion_mode must be in {allowed_fusion_modes}"

        self.fusion_mode = fusion_mode

        self.feat_extractor, n_feats = make_resnet34_feat_extractor()

        if self.fusion_mode == "concat-feats":
            n_feats *= 2

        self.loss_factors = loss_factors
        self.type_weights = type_weights

        # for each abnormality type, we have 3 classes (including absent)
        self.abnorm_clf = nn.ModuleDict()
        for ab in ABNORMALITY_BRANCH_NAMES:
            self.abnorm_clf[ab] = nn.Sequential(
                nn.Linear(n_feats, n_hidden_units),
                nn.ReLU(),
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.ReLU(),
                nn.Linear(n_hidden_units, 3),
            )

        # each tumor is assigned to one of 2 classes (malignant, benign)
        self.type_clf = nn.Sequential(
            nn.Linear(n_feats, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, 1),
        )

    def _compute_losses_on_one_breast(self, batch, logits_type, logits_abnorm):
        losses = {}
        if self.training:
            tgt_type = batch.tgt_type
            if self.fusion_mode != "output":
                tgt_type = tgt_type[0]

            losses["type"] = nn.BCEWithLogitsLoss(reduction="none")(
                logits_type, tgt_type
            )
            losses["type"] *= self.type_weights[batch.meta.classification.values[0]]
            losses["type"] = losses["type"].mean()

            losses["abnorm"] = {}
            for ab in ABNORMALITY_BRANCH_NAMES:
                tgt_abnorm = batch.get_tgt_abnorm(ab).squeeze()
                if self.fusion_mode != "output":
                    tgt_abnorm = tgt_abnorm[0]
                losses["abnorm"][ab] = nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=-1
                )(logits_abnorm[ab], tgt_abnorm)

            losses["abnorm"] = torch.stack(
                [v for v in losses["abnorm"].values()]
            ).mean()

        return losses

    def _forward_backbone(self, images: torch.Tensor):
        feats = self.feat_extractor(images)
        if self.fusion_mode == "mean-feats":
            feats = feats.mean(dim=0)
        elif self.fusion_mode == "max-feats":
            feats = feats.max(dim=0)[0]
        elif self.fusion_mode == "concat-feats":
            feats = torch.cat([f for f in feats])

        return feats

    def _forward_one_breast(self, batch):

        assert (
            batch.meta.breast_id.unique().size == 1
        ), "each forward pass needs a unique breast"

        feats = self._forward_backbone(batch.images)

        logits_type = self.type_clf(feats.squeeze())

        logits_abnorm = {}
        for ab in ABNORMALITY_BRANCH_NAMES:
            logits_abnorm[ab] = self.abnorm_clf[ab](feats.squeeze())

        return logits_type, logits_abnorm

    def forward(self, batch: Batch):
        """
        We experiment on different fusion strategies.

        Modes:
        (1) 'mean'/'max' mode: compute per-breast mean/max feature vector prior
            to classifiers
        (2) 'output' mode: Compute average prediction at the output of classifiers
        """

        out = []
        losses = {"abnorm": [], "type": []}

        for b in batch.groupby("breast_id"):
            logits_type, logits_abnorm = self._forward_one_breast(b)

            l_ = self._compute_losses_on_one_breast(b, logits_type, logits_abnorm)

            if self.training:
                losses["abnorm"].append(l_["abnorm"])
                losses["type"].append(l_["type"])

            # convert logits to predictions
            pred_type = logits_type.sigmoid()
            preds_abnorm = {}
            for ab in ABNORMALITY_BRANCH_NAMES:
                preds_abnorm[ab] = logits_abnorm[ab].softmax(dim=0)

            if self.fusion_mode == "output":
                pred_type = pred_type.mean(dim=0)
                preds_abnorm = {k: v.mean(dim=0) for k, v in preds_abnorm.items()}

            pred_type = pred_type[None, ...]
            preds_abnorm = {k: v[None, ...] for k, v in preds_abnorm.items()}

            n_views = b.get_num_of_views()

            # populate batch fields
            b.set_pred_type(torch.repeat_interleave(pred_type, n_views, dim=0))
            b.set_pred_abnorm(
                calcification=torch.repeat_interleave(
                    preds_abnorm["calcification"], n_views, dim=0
                ),
                mass=torch.repeat_interleave(preds_abnorm["mass"], n_views, dim=0),
            )

            out.append(b)

        if self.training:
            losses = {k: torch.stack(v).nanmean() for k, v in losses.items()}

        return Batch.from_list(out), losses
