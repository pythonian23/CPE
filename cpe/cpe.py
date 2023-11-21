import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import tqdm.notebook
from .callback import Callback


class _Generator(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()


class _Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


class _Encoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()


class CPE:
    def __init__(
        self,
        generator: _Generator,
        discriminator: _Discriminator,
        encoder: _Encoder,
        z_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device: torch.device = device
        self.generator: nn.Module = generator.to(self.device)
        self.discriminator: nn.Module = discriminator.to(self.device)
        self.encoder: nn.Module = encoder.to(self.device)
        self.z_dim: int = z_dim
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def generate_z(self, size: int) -> torch.Tensor:
        return (torch.rand(size, self.z_dim, 1, 1) * 2 - 1).to(self.device)

    def _train_loop(
        self,
        loader: DataLoader,
        train_step: callable,
        epochs: int,
        callback: Callback = None,
        use_tqdm: tuple[int, int] | int = 0,
    ):
        if callback is None:
            callback = Callback()
        if isinstance(use_tqdm, int):
            use_tqdm = (use_tqdm, use_tqdm)

        # Set up TQDM
        epochs_iterator = range(epochs)
        if use_tqdm[0] == 1:
            epochs_iterator = tqdm.tqdm(epochs_iterator, unit="e")
        if use_tqdm[0] == 2:
            epochs_iterator = tqdm.notebook.tqdm_notebook(epochs_iterator, unit="e")
        batches_iterator = range(len(loader))
        if use_tqdm[1] == 1:
            batches_iterator = tqdm.tqdm(batches_iterator, unit="b")
        if use_tqdm[1] == 2:
            batches_iterator = tqdm.notebook.tqdm_notebook(batches_iterator, unit="b")

        callback.on_start(self, epochs_iterator)

        for epoch in epochs_iterator:
            callback.on_epoch_start(epoch, self, epochs_iterator)

            for batch, data in enumerate(loader):
                callback.on_batch_start(epoch, batch, self, epochs_iterator, batches_iterator)
                losses = train_step(data)
                losses = tuple(loss.item() for loss in losses)
                callback.on_batch_end(epoch, batch, self, losses, epochs_iterator, batches_iterator)
                if use_tqdm[1]:
                    batches_iterator.update()

            callback.on_epoch_end(epoch, self, epochs_iterator)
            if use_tqdm[1]:
                batches_iterator.reset()

        callback.on_end(self)

    def train_gan(
        self,
        epochs: int,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        callback: Callback = None,
        use_tqdm: tuple[int, int] | int = 0,
    ):
        def train_step(real):
            real = real[0].to(self.device)

            real_label = torch.ones((real.shape[0], 1), device=self.device)
            fake_label = torch.zeros((real.shape[0], 1), device=self.device)

            z = self.generate_z(real.shape[0])
            d_optimizer.zero_grad()
            real_loss = self.bce(self.discriminator(real), real_label)
            real_loss.backward()
            fake = self.generator(z)
            fake_loss = self.bce(self.discriminator(fake.detach()), fake_label)
            fake_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_loss = self.bce(self.discriminator(fake), real_label)
            g_loss.backward()
            g_optimizer.step()

            return real_loss, fake_loss, g_loss

        self._train_loop(loader, train_step, epochs, callback, use_tqdm)

    def train_encoder(
        self,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        callback: Callback = None,
        use_tqdm: tuple[int, int] | int = 0,
    ):
        self.encoder.train()
        self.generator.eval()

        def train_step(batch):
            batch = batch[0].to(self.device)
            z = self.generate_z(batch.shape[0])

            optimizer.zero_grad()
            e2e_loss = self.mse(self.encoder(self.generator(z).detach()), z)
            e2e_loss.backward()
            i2i_loss = self.mse(self.generator(self.encoder(batch)), batch)
            i2i_loss.backward()
            optimizer.step()

            return e2e_loss, i2i_loss

        self._train_loop(loader, train_step, epochs, callback, use_tqdm)
