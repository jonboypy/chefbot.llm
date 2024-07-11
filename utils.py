# Imports
from pathlib import Path
from types import SimpleNamespace
import yaml
import importlib
from typing import Any
from enum import Enum


class DebugMode(Enum):
    OFF = 1
    FUNCTIONAL = 2
    TRAINING = 3
    FULL = 4


class Config(SimpleNamespace):

    debug: DebugMode = DebugMode.OFF

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
          return Config(**entry)
        return entry

    @staticmethod
    def rev_map_entry(entry):
        if isinstance(entry, Config):
          return entry.to_dict_recursive()
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def to_dict(self) -> dict:
        return vars(self)

    def to_dict_recursive(self) -> dict:
        self = self.to_dict()
        for key, val in self.items():
            if isinstance(val, Config):
                self[key] = val.to_dict_recursive()
            elif isinstance(val, list):
                self[key] = list(map(Config.rev_map_entry, val))
        return self

    def create_instance(self, additional_kwargs: dict=None) -> object:
        cfg = self.to_dict()
        Class = self._get_module_class(next(iter(cfg)))
        kwargs = next(iter(cfg.values())).to_dict()
        if additional_kwargs: kwargs.update(additional_kwargs)
        for k,v in kwargs.items():
            if isinstance(v, str) and '.' in v and '/' not in v:
                kwargs[k] = self._get_module_class(v)
        return Class(**kwargs)

    def _get_module_class(self, spec: str) -> Any:
        module_and_class = spec.split('.')
        clss = module_and_class[-1]
        mod = '.'.join(module_and_class[:-1])
        Class = getattr(importlib.import_module(mod), clss)
        return Class

    def has(self, attr: str) -> bool:
        if (hasattr(self, attr) and
            getattr(self, attr) is not None):
            return True
        else:
            return False

    @staticmethod
    def from_yaml(config_path: Path):
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Config(**config_dict)
        return cfg

    @staticmethod
    def from_yaml_string(yaml_str: str):
        cfg = Config(**yaml.safe_load(yaml_str))
        return cfg
    

