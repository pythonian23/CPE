from torch import nn
import tqdm.notebook


class Callback:
    def on_start(self, model, epochs_iterator: tqdm.notebook.tqdm_notebook | range):
        ...

    def on_epoch_start(self, epoch: int, model, epochs_iterator: tqdm.notebook.tqdm_notebook | range):
        ...

    def on_batch_start(
        self,
        epoch: int,
        batch: int,
        model,
        epochs_iterator: tqdm.notebook.tqdm_notebook | range,
        batches_iterator: tqdm.tqdm | range,
    ):
        ...

    def on_batch_end(
        self,
        epoch: int,
        batch: int,
        model,
        losses: tuple,
        epochs_iterator: tqdm.notebook.tqdm_notebook | range,
        batches_iterator: tqdm.tqdm | range,
    ):
        ...

    def on_epoch_end(self, epoch: int, model, epochs_iterator: tqdm.notebook.tqdm_notebook | range):
        ...

    def on_end(self, model):
        ...


class SaveLossCallback(Callback):
    def __init__(self, lcount: int, bsize: int):
        self.bsize = bsize
        self.buffer = [[] for _ in range(lcount)]
        self.losses = [[] for _ in range(lcount)]

    def _average(self, values):
        return sum(values) / self.bsize

    def on_batch_end(self, epoch, batch, model, losses: tuple, epochs_iterator, batches_iterator):
        for i, loss in enumerate(losses):
            self.buffer[i].append(loss)

        if len(self.buffer[0]) == self.bsize:
            for i, record in enumerate(self.losses):
                record.append(self._average(self.buffer[i]))
            self.buffer = [[] for _ in range(len(self.losses))]


class CallbackList(Callback):
    def __init__(self, *callbacks: Callback):
        self.callbacks = callbacks

    def on_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_start(*args, **kwargs)

    def on_epoch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_start(*args, **kwargs)

    def on_batch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_start(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_end(*args, **kwargs)
