# Imports
from types import SimpleNamespace
from pathlib import Path
import yaml
import copy
import warnings
from typing import Any
import importlib
from enum import Enum


class DebugMode(Enum):
    OFF = 1
    FUNCTIONAL = 2
    TRAINING = 3
    FULL = 4

# Config
class Config(SimpleNamespace):

    debug: DebugMode = DebugMode.OFF
    file: Path = None

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
          return Config(**entry)
        return entry

    @staticmethod
    def rev_map_entry(entry):
        if isinstance(entry, Config):
          return entry.all_dict()
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def dict(self) -> dict:
        # Convert the Config to a dictionary
        return vars(self)
    
    def all_dict(self) -> dict:
        # Convert all Configs contained in this Config to dictionaries
        self = self.dict()
        for key, val in self.items():
            if isinstance(val, Config):
                self[key] = val.all_dict()
            elif isinstance(val, list):
                self[key] = list(map(Config.rev_map_entry, val))
        return self

    def create_instance(self, additional_kwargs: dict=None) -> object:
        # Convert to dictionary
        cfg = self.dict()
        # Load the module class
        Class, init_fn = self._get_module_class(next(iter(cfg)))
        # Get argments to provide to class
        kwargs = copy.deepcopy(next(iter(cfg.values())).dict())
        # Check arguments for classes/instances that need to be loaded
        for k,v in kwargs.items():
            if isinstance(v, str) and '.' in v and '/' not in v:
                kwargs[k] = self._get_module_class(v)
            elif isinstance(v, Config) and '.' in next(iter(v.dict().keys())):
                kwargs[k] = v.create_instance()
            elif isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], str) and '.' in v[i] and '/' not in i:
                        v[i] = self._get_module_class(v[i])
                    elif isinstance(v[i], Config) and '.' in next(iter(v[i].dict().keys())):
                        v[i] = v[i].create_instance()
        # If additional/default arguments were provided from source code...
        if additional_kwargs:
            # Scan the additional arguments
            for k,v in additional_kwargs.items():
                # If they are already defined by user, warn
                if k in kwargs:
                    warnings.warn(f'During creation of {Class} instance, '
                                  f'default argument of "{k}" is being overriden by user. '
                                  'Ensure this is what you intended!')
                # Else update arguments dictionary with additional argument
                else:
                    kwargs[k] = v
        return (Class(**kwargs) if init_fn is None
                else getattr(Class, init_fn)(**kwargs))
    
    def create_all_instances(self) -> None:
        for k,v in self.dict().items():
            if isinstance(v, str) and '.' in v and '/' not in v:
                self.dict()[k] = self._get_module_class(v)
            elif isinstance(v, Config) and '.' in next(iter(v.dict().keys())):
                self.dict()[k] = v.create_instance()
            elif isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], str) and '.' in v[i] and '/' not in i:
                        v[i] = self._get_module_class(v[i])
                    elif isinstance(v[i], Config) and '.' in next(iter(v[i].dict().keys())):
                        v[i] = v[i].create_instance()

    def _get_module_class(self, spec: str) -> Any:
        module_and_class = spec.split('.')
        clss = module_and_class[-1]
        if clss[-2:] == '()':
            init_fn = clss[:-2]
            clss = module_and_class[-2]
            mod = '.'.join(module_and_class[:-2])
            mod = importlib.import_module(mod)
            Class = getattr(mod, clss)
            return Class, init_fn
        else:
            mod = '.'.join(module_and_class[:-1])
            mod = importlib.import_module(mod)
            Class = getattr(mod, clss)
        return Class, None

    def has(self, attr: str) -> bool:
        if (hasattr(self, attr) and
            getattr(self, attr) is not None):
            return True
        else:
            return False
 
    @staticmethod
    def from_yaml(config_file: str):
        config_file = Path(config_file)
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Config(**config_dict, file=config_file)
        return cfg
    
    @staticmethod
    def from_yaml_string(yaml_str: str):
        cfg = Config(**yaml.safe_load(yaml_str))
        return cfg
