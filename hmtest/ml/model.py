from torch import nn
import torch
import torchvision


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

        allowed_fusion_modes = ["features", "predictions"]
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
        (1) 'features' mode: compute per-breast mean feature vector prior
            to classifier
        (2) 'output' mode: Compute average prediction at the output of classifier
        """

        # fuse views with mean aggregation
        all_preds = []
        all_feats = []
        for b in batch.groupby("breast_id"):
            feats = self.feat_extractor(b.images)
            all_feats.append(feats)
            if self.fusion_mode == "features":
                feats = feats.mean(dim=0)
                logits = self.type_clf(feats.squeeze())
                all_preds.append(torch.repeat_interleave(logits.sigmoid(), 2, dim=0))
            else:
                logits = self.type_clf(feats.squeeze())
                preds = logits.sigmoid().mean(dim=0)
                all_preds.append(torch.repeat_interleave(preds, 2, dim=0))

        pred_type = torch.cat(all_preds)[..., None]

        logit_abnorm = self.abnorm_clf(torch.cat(all_feats).squeeze())
        pred_abnorm = logit_abnorm.sigmoid()

        return {
            "abnorm": pred_abnorm,
            "type": pred_type,
        }
