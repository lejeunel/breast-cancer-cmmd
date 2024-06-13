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
        REMINDER: We expect that the channel dimension of each image tensor
        denotes a different view of the same breast.

        We experiment on different fusion strategies thereon.

        TODO: Implement two fusion strategies here.
        """
        images = batch.images

        n_views = images.shape[1]
        views = [torch.unsqueeze(images[:, i, :], 1) for i in range(n_views)]

        feats = torch.cat([self.feat_extractor(view) for view in views], dim=-1)

        # fuse views with mean aggregation
        feats = torch.mean(feats, dim=-1).unsqueeze(-1)
        feats = torch.flatten(feats, 1)

        logit_abnorm = self.abnorm_clf(feats.squeeze())
        logit_type = self.type_clf(feats.squeeze())

        merged = torch.cat((logit_abnorm, logit_type), axis=1)
        logit_type_post = self.type_post_clf(merged)

        return {
            "abnorm": logit_abnorm.sigmoid(),
            "type": logit_type.sigmoid(),
            "type_post": logit_type_post.sigmoid(),
        }
