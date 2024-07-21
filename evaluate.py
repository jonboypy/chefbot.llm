# Imports
import argparse
import torch
from training import LitModule
from utils import Config

parser = argparse.ArgumentParser('Run evaluation.')
parser.add_argument('-cfg', required=True, type=str)
parser.add_argument('-ckpt', required=True, type=str)
args = parser.parse_args()

def main() -> None:
    cfg = Config.from_yaml(args.cfg)
    lm_cfg = cfg.litmodule.dict()['training.LitModule'].cfg
    lm = LitModule.load_from_checkpoint(args.ckpt, cfg=lm_cfg)
    cfg.datamodule.dict()['data.DataModule'].cfg.tokenizer = lm_cfg.tokenizer
    dm = cfg.datamodule.create_instance()
    dm.setup('fit')
    dl = dm.val_dataloader()
    x,y = next(iter(dl))
    start_idx = torch.argwhere(y[0] > -100).min()
    x = x[:1,:start_idx]
    output = lm.net.generate(x.cuda(), max_new_tokens=500)
    print(dm.tokenizer.decode(output[0]))

if __name__ == '__main__':
    main()
