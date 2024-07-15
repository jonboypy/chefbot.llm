# Imports
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from lightning.pytorch import LightningDataModule
from utils import Config


class ChefBotDataset(Dataset):
    """
    A PyTorch Dataset to provide instruction fine-tuning
        data on recipe articles.
    Args:
        data: A list of samples, each a dictionary containing keys:
            ('article', 'task', & 'response')
        tokenizer: A tokenizer for the model being trained.
    """
    def __init__(self, data: list[dict], tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self._data = data
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple:
        article = self._data[index]['article']
        task = self._data[index]['task']
        response = self._data[index]['response']
        prompt = [
            {"role": "system",
             "content": ("You are a kitchen assistant chatbot for a chef. "
                f"The chef wants to make the recipe from this article: {article}")},
            {"role": "user",
             "content": ("Provide only a bulleted list of "
                         f"the recipe's {task}. Do not add any other text.")}]
        x = self._tokenizer.apply_chat_template(
            prompt, return_tensors='pt', add_generation_prompt=True, padding=True)
        y = self._tokenizer.encode(response, return_tensors='pt', padding=True)
        x_len = x.size(-1)
        x = torch.cat([x, y], -1)
        y = torch.cat([torch.tensor([[-100 for _ in range(x_len)]]), y], -1)
        return x, y


class DataModule(LightningDataModule):
    """
    A PTL DataModule that handles loading the data from
        storage, setting up the train & val datasets, etc.
    Args:
        A Config object with settings to provide the DataModule.
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage in ('fit', 'validate', 'test'):
            # (1) Get tokenizer instance
            try: self.tokenizer = self.trainer.lightning_module.tokenizer
            except: self.tokenizer = self.cfg.tokenizer.create_instance()
            # (2) Load dataset from JSON
            with open(self.cfg.data_src) as f:
                data = json.load(f)
            # (3) Re-organize data structure
            data = {k1+'/'+k2: v2 for (k1, v1) in
                    data.items() for (k2, v2) in v1.items()}
            data_rorg = []
            for source, sample in data.items():
                sample1 = {
                    'source': source,
                    'article': sample['article'],
                    'task': 'ingredients',
                    'response': sample['ingredients']}
                sample2 = {
                    'source': source,
                    'article': sample['article'],
                    'task': 'instructions',
                    'response': sample['instructions']}
                data_rorg += [sample1, sample2]
            # (4) Perform train/validation split of data
            train_len = int(len(data_rorg) * self.cfg.train_size)
            if train_len % 2: train_len += 1
            train_data = data_rorg[:train_len]
            val_data = data_rorg[train_len:]
            # (5) Instantiate training & validation datasets
            self.train_ds = self.cfg.dataset.create_instance(
                {'data': train_data, 'tokenizer': self.tokenizer})
            self.val_ds = self.cfg.dataset.create_instance(
                {'data': val_data, 'tokenizer': self.tokenizer})
        else:
            raise NotImplementedError
        
    def _collate_fn(self, samples: list[tuple]) -> tuple[Tensor]:
        x = [sample[0][0] for sample in samples]
        y = [sample[1][0] for sample in samples]
        x = torch.nn.utils.rnn.pad_sequence(
            x, True, self.tokenizer.pad_token_id)
        y = torch.nn.utils.rnn.pad_sequence(y, True, -100)
        return x, y
        
    def train_dataloader(self) -> DataLoader:
        return self.cfg.train_dataloader.create_instance(
            {'dataset': self.train_ds, 'collate_fn': self._collate_fn})

    def val_dataloader(self) -> DataLoader:
        return self.cfg.val_dataloader.create_instance(
            {'dataset': self.val_ds, 'collate_fn': self._collate_fn})

    def test_dataloader(self) -> DataLoader:
        return self.cfg.val_dataloader.create_instance(
            {'dataset': self.val_ds, 'collate_fn': self._collate_fn})
    