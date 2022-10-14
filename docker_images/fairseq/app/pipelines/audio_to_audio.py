import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from app.pipelines import Pipeline
from app.pipelines.utils import ARG_OVERRIDES_MAP
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_speech.hub_interface import S2SHubInterface
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import (
    TTSHubInterface,
    VocoderHubInterface,
)
from huggingface_hub import snapshot_download


class SpeechToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        arg_overrides = ARG_OVERRIDES_MAP.get(
            model_id, {}
        )  # Model specific override. TODO: Update on checkpoint side in the future
        arg_overrides["config_yaml"] = "config.yaml"  # common override
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            model_id,
            arg_overrides=arg_overrides,
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
        self.cfg = cfg
        self.model = models[0].cpu()
        self.model.eval()
        self.task = task

        self.sampling_rate = getattr(self.task, "sr", None) or 16_000

        tgt_lang = self.task.data_cfg.hub.get("tgt_lang", None)
        pfx = f"{tgt_lang}_" if self.task.data_cfg.prepend_tgt_lang_tag else ""

        generation_args = self.task.data_cfg.hub.get(f"{pfx}generation_args", None)
        if generation_args is not None:
            for key in generation_args:
                setattr(cfg.generation, key, generation_args[key])
        self.generator = task.build_generator([self.model], cfg.generation)

        tts_model_id = self.task.data_cfg.hub.get(f"{pfx}tts_model_id", None)
        self.unit_vocoder = self.task.data_cfg.hub.get(f"{pfx}unit_vocoder", None)
        self.tts_model, self.tts_task, self.tts_generator = None, None, None
        if tts_model_id is not None:
            _id = tts_model_id.split(":")[-1]
            cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE")
            if self.unit_vocoder is not None:
                library_name = "fairseq"
                cache_dir = (
                    cache_dir or (Path.home() / ".cache" / library_name).as_posix()
                )
                cache_dir = snapshot_download(
                    f"facebook/{_id}", cache_dir=cache_dir, library_name=library_name
                )

                x = hub_utils.from_pretrained(
                    cache_dir,
                    "model.pt",
                    ".",
                    archive_map=CodeHiFiGANVocoder.hub_models(),
                    config_yaml="config.json",
                    fp16=False,
                    is_vocoder=True,
                )

                with open(f"{x['args']['data']}/config.json") as f:
                    vocoder_cfg = json.load(f)
                assert (
                    len(x["args"]["model_path"]) == 1
                ), "Too many vocoder models in the input"

                vocoder = CodeHiFiGANVocoder(x["args"]["model_path"][0], vocoder_cfg)
                self.tts_model = VocoderHubInterface(vocoder_cfg, vocoder)

            else:
                (
                    tts_models,
                    tts_cfg,
                    self.tts_task,
                ) = load_model_ensemble_and_task_from_hf_hub(
                    f"facebook/{_id}",
                    arg_overrides={"vocoder": "griffin_lim", "fp16": False},
                    cache_dir=cache_dir,
                )
                self.tts_model = tts_models[0].cpu()
                self.tts_model.eval()
                tts_cfg["task"].cpu = True
                TTSHubInterface.update_cfg_with_data_cfg(
                    tts_cfg, self.tts_task.data_cfg
                )
                self.tts_generator = self.tts_task.build_generator(
                    [self.tts_model], tts_cfg
                )

    def __call__(self, inputs: np.array) -> Tuple[np.array, int, List[str]]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default sampled at `self.sampling_rate`.
                The shape of this array is `T`, where `T` is the time axis
        Return:
            A :obj:`tuple` containing:
              - :obj:`np.array`:
                 The return shape of the array must be `C'`x`T'`
              - a :obj:`int`: the sampling rate as an int in Hz.
              - a :obj:`List[str]`: the annotation for each out channel.
                    This can be the name of the instruments for audio source separation
                    or some annotation for speech enhancement. The length must be `C'`.
        """
        _inputs = torch.from_numpy(inputs).unsqueeze(0)
        sample, text = None, None
        if self.cfg.task._name in ["speech_to_text", "speech_to_text_sharded"]:
            sample = S2THubInterface.get_model_input(self.task, _inputs)
            text = S2THubInterface.get_prediction(
                self.task, self.model, self.generator, sample
            )
        elif self.cfg.task._name in ["speech_to_speech"]:
            s2shubinerface = S2SHubInterface(self.cfg, self.task, self.model)
            sample = s2shubinerface.get_model_input(self.task, _inputs)
            text = S2SHubInterface.get_prediction(
                self.task, self.model, self.generator, sample
            )

        wav, sr = np.zeros((0,)), self.sampling_rate
        if self.unit_vocoder is not None:
            tts_sample = self.tts_model.get_model_input(text)
            wav, sr = self.tts_model.get_prediction(tts_sample)
            text = ""
        else:
            tts_sample = TTSHubInterface.get_model_input(self.tts_task, text)
            wav, sr = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_model, self.tts_generator, tts_sample
            )

        return wav.unsqueeze(0).numpy(), sr, [text]
