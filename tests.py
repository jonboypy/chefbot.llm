# Imports
from unittest import TestCase
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from utils import Config

class TestData(TestCase):

    def setUp(self) -> None:
        cfg_yml = """
            datamodule:
              data.DataModule:
                cfg:
                  train_size: 0.5
                  data_src: ./dataset-test-2.json
                  dataset:
                    data.ChefBotDataset: {}
                  train_dataloader:
                    torch.utils.data.DataLoader: {batch_size: 2, shuffle: True}
                  val_dataloader:
                    torch.utils.data.DataLoader: {batch_size: 2, shuffle: False}
                  tokenizer:
                    transformers.AutoTokenizer.from_pretrained():
                      pretrained_model_name_or_path: Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct
        """
        self.cfg = Config.from_yaml_string(cfg_yml)

    def test_datamodule(self) -> None:
        dm = self.cfg.datamodule.create_instance()
        dm.setup('fit')
        self.assertTrue(hasattr(dm, 'tokenizer'))
        self.assertTrue(hasattr(dm, 'train_ds'))
        self.assertTrue(hasattr(dm, 'val_ds'))
        dl = dm.train_dataloader()
        self.assertIsInstance(dl, DataLoader)
        x, y = next(iter(dl))
        self.assertIsInstance(x, Tensor)
        self.assertIsInstance(y, Tensor)
        self.assertTrue(x.dtype == torch.int64)
        self.assertTrue(y.dtype == torch.int64)

class TestTraining(TestCase):

    def setUp(self) -> None:
        cfg_yml = """
            litmodule:
              training.LitModule:
                cfg:
                  tokenizer:
                    transformers.AutoTokenizer.from_pretrained():
                      pretrained_model_name_or_path: Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct
                  network:
                    transformers.AutoModelForCausalLM.from_pretrained():
                      pretrained_model_name_or_path: Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct
                      torch_dtype: bfloat16
                      device_map: auto
                  lora:
                    r: 8
                    inference_mode: False
                    lora_alpha: 32
                    lora_dropout: 0.1
        """
        self.cfg = Config.from_yaml_string(cfg_yml)

    def test_litmodule(self) -> None:
        lm = self.cfg.litmodule.create_instance()
        self.assertTrue(hasattr(lm, 'net'))

class TestRunner(TestCase):
    ...
