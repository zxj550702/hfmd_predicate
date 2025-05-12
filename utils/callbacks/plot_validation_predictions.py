
from utils.callbacks.base import BestEpochCallback
import utils


class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        predictions, y = outputs
        if len(self.predictions) > 0:
            last_r2 = utils.metrics.r2(self.predictions[0], self.ground_truths[0]).cpu().numpy()
            current_r2 = utils.metrics.r2(predictions, y).cpu().numpy()
            if (current_r2 < last_r2):
                 return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y)
        self.predictions.append(predictions)

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
