# Imports
import lightning.pytorch as ptl
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from utils import Config
from torch import Tensor


class LitModule(ptl.LightningModule):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct")
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                 inference_mode=False, r=8,
                                 lora_alpha=32, lora_dropout=0.1)
        self.model = get_peft_model(model, peft_config)
        
    def training_step(self, batch: tuple[Tensor]) -> dict[str, Tensor]:
        ...

