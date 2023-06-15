import json
import logging
import os

import torch
from app import idle, timing
from app.pipelines import Pipeline
from huggingface_hub import hf_hub_download, model_info
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
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

    def _process_req(self, inputs: str, **kwargs) -> str:
        """
        Args:
            inputs (:obj:`str`):
                a string for text to be completed
        Returns:
            A string of completed text.
        """
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt")
        self._model_to_gpu()

        if torch.cuda.is_available():
            device = "cuda"
            tokenized_inputs = {
                "input_ids": tokenized_inputs["input_ids"].to(device),
                "attention_mask": tokenized_inputs["attention_mask"].to(device),
            }
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=tokenized_inputs["input_ids"],
                attention_mask=tokenized_inputs["attention_mask"],
                max_new_tokens=10,
                eos_token_id=3,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
