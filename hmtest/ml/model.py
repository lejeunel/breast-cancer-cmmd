from torch import nn
import torch
import torchvision


def make_resnet18_feat_extractor() -> tuple[nn.Module, int]:
    """
    Replace input layer with a single-channel conv2d,
    remove the last classification layer,
    and extract the dimensions of the feature vector
    """
    rn = torchvision.models.resnet18()
    feat_extractor = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        rn.bn1,
        rn.relu,
        rn.maxpool,
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
    ):
        super().__init__()

        self.feat_extractor, n_feats = make_resnet18_feat_extractor()
        self.abnorm_clf = nn.Sequential(
            nn.Linear(n_feats, n_abnormalities - 1),
        )

        self.type_clf = nn.Sequential(
            nn.Linear(n_feats, n_types - 1),
        )

        self.type_post_clf = nn.Sequential(
            nn.Linear(n_abnormalities + n_types - 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, batch):
        images = batch.images

        feats = self.feat_extractor(images)
        feats = torch.flatten(feats, 1)

        logit_abnorm = self.abnorm_clf(feats.squeeze())
        logit_type = self.type_clf(feats.squeeze())

        merged = torch.cat((logit_abnorm, logit_type), axis=1)
        logit_type_post = self.type_post_clf(merged)

        return {
            "abnorm": logit_abnorm,
            "type": logit_type,
            "type_post": logit_type_post,
        }
