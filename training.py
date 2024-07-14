# Imports
from typing import Any
import lightning.pytorch as ptl
from peft import LoraConfig, TaskType, get_peft_model
from utils import Config
from torch import Tensor


class LitModule(ptl.LightningModule):
    """
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer.create_instance()
        self.net = cfg.network.create_instance()
        if cfg.has('lora'):
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                     **self.cfg.lora.to_dict())
            self.net = get_peft_model(self.net, peft_config)

    def configure_optimizers(self):
        opt = self.cfg.optimizer.create_instance(
            {'params': self.net.parameters()})
        if self.cfg.has('lr_scheduler'):
            lr_sched = self.cfg.lr_scheduler.create_instance()
            return opt, lr_sched
        else:
            return opt
 
    def configure_callbacks(self) -> ptl.Callback:
        return super().configure_callbacks()

    def training_step(self, batch: tuple[Tensor]) -> dict[str, Tensor]:
        results = {}
        x, y = batch
        output = self.net(input_ids=x, labels=y)
        {'loss': output.loss, 'training_loss': output.loss}
        return results

    def validation_step(self, batch: tuple[Tensor]) -> dict[str, Tensor]:
        results = {}
        x, y = batch
        output = self.net(input_ids=x, labels=y)
        {'validation_loss': output.loss}
        return results

    def test_step(self, batch: tuple[Tensor]) -> dict[str, Tensor]:
        results = {}
        x, y = batch
        output = self.net(input_ids=x, labels=y)
        {'validation_loss': output.loss}
        return results

    class LoggingCallback(ptl.Callback):
        """
        PTL Callback that handles logging to keep training code clean.
        """
        def __init__(self) -> None:
            super().__init__()

        def on_train_batch_end(self, trainer: ptl.Trainer,
                               pl_module: ptl.LightningModule,
                               outputs: dict, batch: tuple[Tensor],
                               batch_idx: int) -> None:
            del outputs['loss']
            for name, metric in outputs.items():
                pl_module.log(name, metric)

        def on_validation_batch_end(self, trainer: ptl.Trainer,
                                    pl_module: ptl.LightningModule,
                                    outputs: dict, batch: tuple, batch_idx: int,
                                    dataloader_idx: int = 0) -> None:
            for name, metric in outputs.items():
                pl_module.log(name, metric)

        def on_test_batch_end(self, trainer: ptl.Trainer,
                              pl_module: ptl.LightningModule,
                              outputs: dict, batch: tuple, batch_idx: int,
                              dataloader_idx: int = 0) -> None:
            for name, metric in outputs.items():
                pl_module.log(name, metric)



