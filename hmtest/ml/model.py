from torch import nn
import torch
import torchvision
from hmtest.ml.dataloader import Batch


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


class BreastClassifier(nn.Module):
    def __init__(
        self,
        n_abnormalities: int = 2,
        n_types: int = 2,
        dropout_rate: float = 0.0,
        fusion_mode="features",
    ):
        super().__init__()

        allowed_fusion_modes = ["mean-feats", "max-feats", "output"]
        assert (
            fusion_mode in allowed_fusion_modes
        ), f"fusion_mode must be in {allowed_fusion_modes}"

        self.fusion_mode = fusion_mode

        self.feat_extractor, n_feats = make_resnet34_feat_extractor()
        self.abnorm_clf = nn.Sequential(
            nn.Linear(n_feats, n_abnormalities),
        )

        self.type_clf = nn.Sequential(
            nn.Linear(n_feats, n_types - 1),
        )

        self.type_post_clf = nn.Sequential(
            nn.Linear(n_abnormalities + n_types - 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, batch):
        """
        We experiment on different fusion strategies.

        Modes:
        (1) 'mean'/'max' mode: compute per-breast mean/max feature vector prior
            to classifiers
        (2) 'output' mode: Compute average prediction at the output of classifiers
        """

        out = []

        for b in batch.groupby("breast_id"):
            n_views = b.get_num_of_views()
            feats = self.feat_extractor(b.images)
            if self.fusion_mode == "mean-feats":
                feats = feats.mean(dim=0)
            elif self.fusion_mode == "max-feats":
                feats = feats.max(dim=0)[0]
            logits_type = self.type_clf(feats.squeeze())
            logits_abnorm = self.abnorm_clf(feats.squeeze())

            preds_type = logits_type.sigmoid()
            preds_abnorm = logits_abnorm.sigmoid()

            if self.fusion_mode == "output":
                preds_type = preds_type.mean(dim=0)
                preds_abnorm = preds_abnorm.mean(dim=0)

            preds_type = preds_type[None, ...]
            preds_abnorm = preds_abnorm[None, ...]

            b.set_predictions(
                torch.repeat_interleave(preds_type, n_views, dim=0), "type"
            )
            b.set_predictions(
                torch.repeat_interleave(preds_abnorm, n_views, dim=0), "abnorm"
            )

            out.append(b)

        return Batch.from_list(out)
