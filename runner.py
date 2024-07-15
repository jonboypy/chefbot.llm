# Imports
import argparse
from utils import Config

class Runner:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lm = cfg.litmodule.create_instance()
        self.dm = cfg.datamodule.create_instance()
        self.trainer = cfg.trainer.create_instance()

    def execute(self, routine: str) -> None:
        if routine == 'train':
            self.trainer.fit(self.lm, self.dm)
        elif routine == 'validate':
            self.trainer.validate(self.lm, self.dm)
        elif routine == 'test':
            self.trainer.test(self.lm, self.dm)
        elif routine == 'predict':
            self.trainer.test(self.lm, self.dm)
        else:
            raise NotImplementedError(routine)

# CLI
parser = argparse.ArgumentParser('Run various routines of the training pipeline.')
parser.add_argument('-cfg', default='cfg.yml', type=str)
parser.add_argument('-routine', required=True, type=str)
args = parser.parse_args()

def main() -> None:
    cfg = Config.from_yaml(args.cfg)
    runner = Runner(cfg)
    runner.execute(args.routine)

if __name__ == '__main__':
    main()
