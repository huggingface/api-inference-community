from app.pipelines import Pipeline
from peft import (
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from app import idle, timing
import torch
from huggingface_hub import hf_hub_download, model_info

import json
import os
import logging

logger = logging.getLogger(__name__)

class TextGenerationPipeline(Pipeline):
    def __init__(self, model_id: str):
        use_auth_token = os.getenv("HF_API_TOKEN")
        model_data = model_info(model_id, token=use_auth_token)

        has_config = any(
            file.rfilename == "adapter_config.json" for file in model_data.siblings
        )

        if has_config:
            config_file = hf_hub_download(
                model_id, "adapter_config.json", token=use_auth_token
            )
            with open(config_file, "r") as f:
                config_dict = json.load(f)
        else:
            raise FileNotFoundError("Config file is not found in model repository")

        base_model_id = config_dict["base_model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", trust_remote_code=True
        )
        # wrap base model with peft
        self.model = PeftModel.from_pretrained(model, model_id)

    
    def __call__(self, inputs: str, **kwargs) -> str:
        """
        Args:
            inputs (:obj:`str`):
                a string for text to be completed
        Returns:
            A string of completed text.
        """
        if idle.UNLOAD_IDLE:
            with idle.request_witnesses():
                self._model_to_gpu()
                resp = self._process_req(inputs, **kwargs)
        else:
            resp = self._process_req(inputs, **kwargs)
        return resp

    @timing.timing
    def _model_to_gpu(self):
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"


    def _process_req(self, inputs: str, **kwargs) -> str:
        """
        Args:
            inputs (:obj:`str`):
                a string for text to be completed
        Returns:
            A string of completed text.
        """
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
            )
        
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)