import logging

from huggingface_hub import model_info


logger = logging.getLogger(__name__)


class LoRAPipelineMixin(object):
    def __init__(self):
        if not hasattr(self, "current_lora_adapter"):
            self.current_lora_adapter = None
        if not hasattr(self, "model_id"):
            self.model_id = None

    @staticmethod
    def _get_lora_weight_name(model_data):
        is_diffusers_lora = LoRAPipelineMixin._is_diffusers_lora(model_data)
        file_to_load = next(
            (
                file.rfilename
                for file in model_data.siblings
                if file.rfilename.endswith(".safetensors")
            ),
            None,
        )
        if not file_to_load and not is_diffusers_lora:
            raise ValueError("No *.safetensors file found for your LoRA")
        weight_name = file_to_load if not is_diffusers_lora else None
        return weight_name

    @staticmethod
    def _is_lora(model_data):
        return LoRAPipelineMixin._is_diffusers_lora(
            model_data
        ) or "lora" in model_data.cardData.get("tags", [])

    @staticmethod
    def _is_diffusers_lora(model_data):
        is_diffusers_lora = any(
            file.rfilename
            in ("pytorch_lora_weights.bin", "pytorch_lora_weights.safetensors")
            for file in model_data.siblings
        )
        return is_diffusers_lora

    def _load_lora_adapter(self, kwargs):
        adapter = kwargs.pop("lora_adapter", None)
        if adapter is not None:
            logger.info("LoRA adapter %s requested", adapter)
            if adapter != self.current_lora_adapter:
                model_data = model_info(adapter, token=self.use_auth_token)
                if not self._is_lora(model_data):
                    msg = f"Requested adapter {adapter:s} is not a LoRA adapter"
                    logger.error(msg)
                    raise ValueError(msg)
                base_model = model_data.cardData["base_model"]
                if self.model_id != base_model:
                    msg = f"Requested adapter {adapter:s} is not a LoRA adapter for base model {self.model_id:s}"
                    logger.error(msg)
                    raise ValueError(msg)
                logger.info(
                    "LoRA adapter %s needs to be replaced with compatible adapter %s",
                    self.current_lora_adapter,
                    adapter,
                )
                self.current_lora_adapter = None
                self.ldm.unload_lora_weights()
                logger.info("LoRA weights unloaded, loading new weights")
                weight_name = self._get_lora_weight_name(model_data=model_data)
                self.ldm.load_lora_weights(
                    adapter, weight_name=weight_name, use_auth_token=self.use_auth_token
                )
                self.current_lora_adapter = adapter
                logger.info("LoRA weights loaded for adapter %s", adapter)
            else:
                logger.info("LoRA adapter %s already loaded", adapter)
        elif self.current_lora_adapter is not None:
            logger.info(
                "No LoRA adapter requested, unloading weights and using base model %s",
                self.model_id,
            )
            self.ldm.unload_lora_weights()
            self.current_lora_adapter = None
