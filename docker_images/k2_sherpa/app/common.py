import functools
import json
from typing import List, Optional, Union

import k2
import kaldifeat
import sentencepiece as spm
import torch
from huggingface_hub import HfApi, hf_hub_download
from sherpa import RnntConformerModel

from .decode import (
    run_model_and_do_greedy_search,
    run_model_and_do_modified_beam_search,
)


def get_hfconfig(model_id, config_name="hf_demo"):
    info = HfApi().model_info(repo_id=model_id)
    config_file = hf_hub_download(model_id, filename="config.json")
    with open(config_file) as config:
        info.config = json.load(config)

    if info.config and config_name is not None:
        if config_name in info.config:
            return info.config[config_name]
        else:
            raise ValueError("Config section " + config_name + " not found")
    else:
        return info


def model_from_hfconfig(hf_repo, hf_config):
    nn_model_filename = hf_hub_download(hf_repo, hf_config["nn_model_filename"])
    token_filename = (
        hf_hub_download(hf_repo, hf_config["token_filename"])
        if "token_filename" in hf_config
        else None
    )
    bpe_model_filename = (
        hf_hub_download(hf_repo, hf_config["bpe_model_filename"])
        if "bpe_model_filename" in hf_config
        else None
    )
    decoding_method = hf_config.get("decoding_method", "greedy_search")
    sample_rate = hf_config.get("sample_rate", 16000)
    num_active_paths = hf_config.get("num_active_paths", 4)

    assert decoding_method in ("greedy_search", "modified_beam_search"), decoding_method

    if decoding_method == "modified_beam_search":
        assert num_active_paths >= 1, num_active_paths

    assert bpe_model_filename is not None or token_filename is not None
    if bpe_model_filename:
        assert token_filename is None

    if token_filename:
        assert bpe_model_filename is None

    return OfflineAsr(
        nn_model_filename,
        bpe_model_filename,
        token_filename,
        decoding_method,
        num_active_paths,
        sample_rate,
    )


def transcribe_batch_from_tensor(model, batch):
    return model.decode_waves([batch])[0]


class OfflineAsr(object):
    def __init__(
        self,
        nn_model_filename: str,
        bpe_model_filename: Optional[str],
        token_filename: Optional[str],
        decoding_method: str,
        num_active_paths: int,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
          nn_model_filename:
            Path to the torch script model.
          bpe_model_filename:
            Path to the BPE model. If it is None, you have to provide
            `token_filename`.
          token_filename:
            Path to tokens.txt. If it is None, you have to provide
            `bpe_model_filename`.
          decoding_method:
            The decoding method to use. Currently, only greedy_search and
            modified_beam_search are implemented.
          num_active_paths:
            Used only when decoding_method is modified_beam_search.
            It specifies number of active paths for each utterance. Due to
            merging paths with identical token sequences, the actual number
            may be less than "num_active_paths".
          sample_rate:
            Expected sample rate of the feature extractor.
          device:
            The device to use for computation.
        """
        self.model = RnntConformerModel(
            filename=nn_model_filename,
            device=device,
            optimize_for_inference=False,
        )

        if bpe_model_filename:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model_filename)
        else:
            self.token_table = k2.SymbolTable.from_file(token_filename)

        self.sample_rate = sample_rate
        self.feature_extractor = self._build_feature_extractor(
            sample_rate=sample_rate,
            device=device,
        )

        assert decoding_method in (
            "greedy_search",
            "modified_beam_search",
        ), decoding_method
        if decoding_method == "greedy_search":
            nn_and_decoding_func = run_model_and_do_greedy_search
        elif decoding_method == "modified_beam_search":
            nn_and_decoding_func = functools.partial(
                run_model_and_do_modified_beam_search,
                num_active_paths=num_active_paths,
            )
        else:
            raise ValueError(
                f"Unsupported decoding_method: {decoding_method} "
                "Please use greedy_search or modified_beam_search"
            )

        self.nn_and_decoding_func = nn_and_decoding_func
        self.device = device

    def _build_feature_extractor(
        self,
        sample_rate: int = 16000,
        device: Union[str, torch.device] = "cpu",
    ) -> kaldifeat.OfflineFeature:
        """Build a fbank feature extractor for extracting features.

        Args:
          sample_rate:
            Expected sample rate of the feature extractor.
          device:
            The device to use for computation.
        Returns:
          Return a fbank feature extractor.
        """
        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = 80

        fbank = kaldifeat.Fbank(opts)

        return fbank

    def decode_waves(self, waves: List[torch.Tensor]) -> List[List[str]]:
        """
        Args:
          waves:
            A list of 1-D torch.float32 tensors containing audio samples.
            wavs[i] contains audio samples for the i-th utterance.

            Note:
              Whether it should be in the range [-32768, 32767] or be normalized
              to [-1, 1] depends on which range you used for your training data.
              For instance, if your training data used [-32768, 32767],
              then the given waves have to contain samples in this range.

              All models trained in icefall use the normalized range [-1, 1].
        Returns:
          Return a list of decoded results. `ans[i]` contains the decoded
          results for `wavs[i]`.
        """
        waves = [w.to(self.device) for w in waves]
        features = self.feature_extractor(waves)

        tokens = self.nn_and_decoding_func(self.model, features)

        if hasattr(self, "sp"):
            results = self.sp.decode(tokens)
        else:
            results = [[self.token_table[i] for i in hyp] for hyp in tokens]
            results = ["".join(r) for r in results]

        return results
