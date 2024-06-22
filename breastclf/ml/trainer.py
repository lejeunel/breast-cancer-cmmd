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
        device=torch.device("cpu"),
        loss_factors={"type": 1, "abnorm": 0},
        start_epoch=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.train_epoch = start_epoch
        self.val_epoch = start_epoch

        self.loss_factors = loss_factors

    def train_one_epoch(self, dataloader, callbacks=[]):

        self.model.train()
        self.model.to(self.device)

        iter = 0
        for batch in (pbar := tqdm(dataloader)):
            batch.to(self.device)

            self.optimizer.zero_grad()

            batch, losses = self.model(batch)

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

            batch.iter = self.train_epoch * len(dataloader) + iter

            for clbk in callbacks:
                clbk.on_batch_end(batch)

            iter += 1

        for clbk in callbacks:
            clbk.on_epoch_end(epoch=self.train_epoch)

        self.train_epoch += 1

    @torch.no_grad
    def eval(self, dataloader, callbacks=[]):

        self.model.eval()
        self.model.to(self.device)

        iter = 0
        for batch in (pbar := tqdm(dataloader)):

            batch.to(self.device)
            batch, losses = self.model(batch)

            pbar.set_description("[val]")

            batch.iter = self.val_epoch * len(dataloader) + iter

            for clbk in callbacks:
                clbk.on_batch_end(batch)

            iter += 1

        for clbk in callbacks:
            clbk.on_epoch_end(epoch=self.val_epoch)

        self.val_epoch += 1
