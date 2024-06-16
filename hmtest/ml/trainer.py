import torch
from tqdm import tqdm


class Trainer:
    """
    Wrapper class used in training and validation routine
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        loss_factors={"type": 1, "abnorm": 1},
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

    def _compute_losses(self, batch):
        losses = {}
        losses["type"] = self.criterion(batch.pred_type, batch.tgt_type)
        losses["type"] = self._weigh_type_loss(losses["type"], batch.tgt_type)
        losses["type"] = losses["type"].mean()

        losses["abnorm"] = self.criterion(batch.pred_abnorm, batch.tgt_abnorm).mean()

        return losses

    def train_one_epoch(self, dataloader, callbacks=[]):

        self.model.train()
        self.model.to(self.device)

        for batch in (pbar := tqdm(dataloader)):
            batch.to(self.device)

            self.optimizer.zero_grad()

            batch = self.model(batch)

            losses = self._compute_losses(batch)

            total_loss = 0
            for k, v in self.loss_factors.items():
                losses[k] = v * losses[k]
                total_loss += losses[k]

            total_loss.backward()
            self.optimizer.step()

            for k, v in losses.items():
                batch.set_loss(v, k)

            pbar.set_description(
                f"[train] lss: {total_loss.detach().cpu().numpy().sum():.2e}"
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

            batch.to(self.device)
            batch = self.model(batch)
            losses = self._compute_losses(batch)

            for k, v in losses.items():
                batch.set_loss(v, k)

            pbar.set_description("[val]")

            batch.iter = self.val_iter

            for clbk in callbacks:
                clbk.on_batch_end(batch)

            self.val_iter += 1

        for clbk in callbacks:
            clbk.on_epoch_end(epoch=self.val_epoch)

        self.val_epoch += 1
