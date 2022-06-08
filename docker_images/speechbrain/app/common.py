from enum import Enum

from huggingface_hub import HfApi


class ModelType(Enum):
    # audio-to-audio
    SEPFORMERSEPARATION = "SEPFORMERSEPARATION"
    SPECTRALMASKENHANCEMENT = "SPECTRALMASKENHANCEMENT"
    # automatic-speech-recognition
    ENCODERASR = "ENCODERASR"
    ENCODERDECODERASR = "ENCODERDECODERASR"
    # audio-clasification
    ENCODERCLASSIFIER = "ENCODERCLASSIFIER"
    # text-to-speech
    TACOTRON2 = "TACOTRON2"
    HIFIGAN = "HIFIGAN"


def get_type(model_id, interface_type="interface"):
    info = HfApi().model_info(repo_id=model_id)
    if info.config:
        if "speechbrain" in info.config:
            if interface_type in info.config["speechbrain"]:
                return ModelType(info.config["speechbrain"][interface_type].upper())
            else:
                raise ValueError(f"{interface_type} not in config.json")
        else:
            raise ValueError("speechbrain_interface not in config.json")
    raise ValueError("no config.json in repository")


def get_vocoder_model_id(model_id):
    info = HfApi().model_info(repo_id=model_id)
    if info.config:
        if "speechbrain" in info.config:
            if "vocoder_model_id" in info.config["speechbrain"]:
                return info.config["speechbrain"]["vocoder_model_id"]
            else:
                raise ValueError("vocoder_model_id not in config.json")
        else:
            raise ValueError("speechbrain_interface not in config.json")
    raise ValueError("no config.json in repository")
