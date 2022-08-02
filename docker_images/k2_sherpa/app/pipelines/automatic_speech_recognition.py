from typing import Dict

import app.common as cx
import numpy as np
import torch
from app.pipelines import Pipeline


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


class AutomaticSpeechRecognitionPipeline(Pipeline):
    def __init__(self, model_id: str):
        model_config = cx.get_hfconfig(model_id, "hf_demo")
        self.model = cx.model_from_hfconfig(hf_repo=model_id, hf_config=model_config)

        self.sampling_rate = self.model.sample_rate

    def __call__(self, inputs: np.array) -> Dict[str, str]:
        """
        Args:
            inputs (:obj:`np.array`):
                The raw waveform of audio received. By default at self.sampling_rate, otherwise 16KHz.
                Check `app.validation` if a different sample rate is required
                or if it depends on the model
        Return:
            A :obj:`dict`:. The object return should be liked {"text": "XXX"} containing
            the detected langage from the input audio
        """
        batch = torch.from_numpy(inputs)
        words = cx.transcribe_batch_from_tensor(self.model, batch)
        return {"text": words}
