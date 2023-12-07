import logging

import torch.nn as nn
from huggingface_hub import hf_hub_download, model_info
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


class LoRAPipelineMixin(object):
    def __init__(self):
        if not hasattr(self, "current_lora_adapter"):
            self.current_lora_adapter = None
        if not hasattr(self, "model_id"):
            self.model_id = None
        if not hasattr(self, "current_tokens_loaded"):
            self.current_tokens_loaded = 0

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

    @staticmethod
    def _is_safetensors_pivotal(model_data):
        embeddings_safetensors_exists = any(
            sibling.rfilename == "embeddings.safetensors"
            for sibling in model_data.siblings
        )
        return embeddings_safetensors_exists

    @staticmethod
    def _is_pivotal_tuning_lora(model_data):
        return LoRAPipelineMixin._is_safetensors_pivotal(model_data) or any(
            sibling.rfilename == "embeddings.pti" for sibling in model_data.siblings
        )

    def _fuse_or_raise(self):
        try:
            self.ldm.fuse_lora(safe_fusing=True)
        except ValueError as e:
            logger.exception(e)
            logger.warning("Unable to fuse LoRA adapter")
            self.ldm.unload_lora_weights()
            self.current_lora_adapter = None
            raise

    def _reset_tokenizer_and_encoder(self, tokenizer, text_encoder, token_to_remove):
        token_id = tokenizer(token_to_remove)["input_ids"][1]
        del tokenizer._added_tokens_decoder[token_id]
        del tokenizer._added_tokens_encoder[token_to_remove]
        tokenizer._update_trie()

        tokenizer_size = len(tokenizer)
        text_embedding_dim = text_encoder.get_input_embeddings().embedding_dim
        text_embedding_weights = text_encoder.get_input_embeddings().weight[
            :tokenizer_size
        ]
        text_embeddings_filtered = nn.Embedding(tokenizer_size, text_embedding_dim)
        text_embeddings_filtered.weight.data = text_embedding_weights
        text_encoder.set_input_embeddings(text_embeddings_filtered)

    def _unload_textual_embeddings(self):
        if self.current_tokens_loaded > 0:
            for i in range(self.current_tokens_loaded):
                token_to_remove = f"<s{i}>"
                self._reset_tokenizer_and_encoder(
                    self.ldm.tokenizer, self.ldm.text_encoder, token_to_remove
                )
                self._reset_tokenizer_and_encoder(
                    self.ldm.tokenizer_2, self.ldm.text_encoder_2, token_to_remove
                )
        self.current_tokens_loaded = 0
    
    def _load_textual_embeddings(self, adapter, model_data):
        if self._is_pivotal_tuning_lora(model_data):
            embedding_path = hf_hub_download(
                repo_id=adapter,
                filename="embeddings.safetensors"
                if self._is_safetensors_pivotal(model_data)
                else "embeddings.pti",
                repo_type="model",
            )
            embeddings = load_file(embedding_path)
            state_dict_clip_l = (
                embeddings.get("text_encoders_0")
                if "text_encoders_0" in embeddings
                else embeddings.get("clip_l", None)
            )
            state_dict_clip_g = (
                embeddings.get("text_encoders_1")
                if "text_encoders_1" in embeddings
                else embeddings.get("clip_g", None)
            )
            tokens_to_add = 0 if state_dict_clip_l is None else len(state_dict_clip_l)
            tokens_to_add_2 = 0 if state_dict_clip_g is None else len(state_dict_clip_g)
            if tokens_to_add == tokens_to_add_2 and tokens_to_add > 0:
                if state_dict_clip_l is not None and len(state_dict_clip_l) > 0:
                    token_list = [f"<s{i}>" for i in range(tokens_to_add)]
                    self.ldm.load_textual_inversion(
                        state_dict_clip_l,
                        token=token_list,
                        text_encoder=self.ldm.text_encoder,
                        tokenizer=self.ldm.tokenizer,
                    )

                if state_dict_clip_g is not None and len(state_dict_clip_g) > 0:
                    token_list = [f"<s{i}>" for i in range(tokens_to_add_2)]
                    self.ldm.load_textual_inversion(
                        state_dict_clip_g,
                        token=token_list,
                        text_encoder=self.ldm.text_encoder_2,
                        tokenizer=self.ldm.tokenizer_2,
                    )
                logger.info("Text embeddings loaded for adapter %s", adapter)
            else:
                logger.info(
                    "No text embeddings were loaded due to invalid embeddings or a mismatch of token sizes for adapter %s",
                    adapter,
                )
            self.current_tokens_loaded = tokens_to_add

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
                self.ldm.unfuse_lora()
                self.ldm.unload_lora_weights()
                self._unload_textual_embeddings()
                self.current_lora_adapter = None
                logger.info("LoRA weights unloaded, loading new weights")
                weight_name = self._get_lora_weight_name(model_data=model_data)

                self.ldm.load_lora_weights(
                    adapter, weight_name=weight_name, use_auth_token=self.use_auth_token
                )
                self.current_lora_adapter = adapter
                self._fuse_or_raise()
                logger.info("LoRA weights loaded for adapter %s", adapter)
                self._load_textual_embeddings(adapter, model_data)
            else:
                logger.info("LoRA adapter %s already loaded", adapter)
                # Needed while a LoRA is loaded w/ model
                model_data = model_info(adapter, token=self.use_auth_token)
                if (
                    self._is_pivotal_tuning_lora(model_data)
                    and self.current_tokens_loaded == 0
                ):
                    self._load_textual_embeddings(adapter, model_data)
        elif self.current_lora_adapter is not None:
            logger.info(
                "No LoRA adapter requested, unloading weights and using base model %s",
                self.model_id,
            )
            self.ldm.unfuse_lora()
            self.ldm.unload_lora_weights()
            self._unload_textual_embeddings()
            self.current_lora_adapter = None
            self.current_tokens_loaded = 0
