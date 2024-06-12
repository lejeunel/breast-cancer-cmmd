from tqdm import tqdm
import torch
from torchvision.ops import sigmoid_focal_loss


class Trainer:
    """
    Wrapper class used in training and validation routine
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loss_factors={"type": 1, "type_post": 0, "abnorm": 0},
        device=torch.device("cpu"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.loss_factors = loss_factors

        self.train_epoch = 1
        self.train_iter = 0
        self.val_epoch = 1
        self.val_iter = 0

    def _weigh_type_loss(self, loss, target, freq_pos=0.7):
        loss[target == 1] = loss[target == 1] / freq_pos
        loss[target == 0] = loss[target == 0] / (1 - freq_pos)
        return loss

    def _compute_losses(self, batch, logits):
        losses = {}
        losses["type"] = self.criterion(logits["type"], batch.tgt_type)
        losses["type"] = losses["type"].mean()

        losses["type_post"] = self.criterion(logits["type_post"], batch.tgt_type)
        losses["type_post"] = losses["type_post"].mean()

        losses["abnorm"] = self.criterion(logits["abnorm"], batch.tgt_abnorm).mean()

        return losses

    def train_one_epoch(self, dataloader, callbacks=[]):

        self.model.train()
        self.model.to(self.device)

        for batch in (pbar := tqdm(dataloader)):

            logits = self.model(batch)

            self.optimizer.zero_grad()
            losses = self._compute_losses(batch, logits)

            total_loss = 0
            for k, v in self.loss_factors.items():
                losses[k] = v * losses[k]
                total_loss += losses[k]

            total_loss.backward()
            self.optimizer.step()

            batch.set_losses(losses)
            batch.set_predictions(logits)

            pbar.set_description(
                f"[train] lss: {total_loss.detach().numpy().sum():.2e}"
            )

            batch.iter = self.train_iter

            for clbk in callbacks:
                clbk.on_batch_end(batch)

            self.train_iter += 1

        for clbk in callbacks:
            clbk.on_epoch_end(epoch=self.train_epoch)

        self.train_epoch += 1

    @torch.no_grad
    def eval(self, dataloader, callbacks=[]):

        self.model.eval()
        self.model.to(self.device)

        for batch in (pbar := tqdm(dataloader)):

            logits = self.model(batch)
            losses = self._compute_losses(batch, logits)

            batch.set_losses(losses)
            batch.set_predictions(logits)

            pbar.set_description("[val]")

            batch.iter = self.val_iter

            for clbk in callbacks:
                clbk.on_batch_end(batch)

            self.val_iter += 1

        for clbk in callbacks:
            clbk.on_epoch_end(epoch=self.val_epoch)

        self.val_epoch += 1
