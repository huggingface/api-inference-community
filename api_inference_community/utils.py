from api_inference_community.constants import AUDIO_INPUTS, IMAGE_INPUTS


def get_metric(inputs, task, pipe) -> dict[str, int | float]:
    if task in AUDIO_INPUTS:
        return {"x-compute-audio-length": get_audio_length(inputs, pipe.sampling_rate)}
    elif task in IMAGE_INPUTS:
        return {"x-compute-images": 1}
    else:
        return {"x-compute-characters": get_input_characters(inputs)}


def get_audio_length(inputs, sampling_rate: int) -> float:
    if isinstance(inputs, dict):
        # Should only apply for internal AsrLive
        length_in_s = inputs["raw"].shape[0] / inputs["sampling_rate"]
    else:
        length_in_s = inputs.shape[0] / sampling_rate
    return length_in_s


def get_input_characters(inputs) -> int:
    if isinstance(inputs, str):
        return len(inputs)
    elif isinstance(inputs, (tuple, list)):
        return sum(get_input_characters(input_) for input_ in inputs)
    elif isinstance(inputs, dict):
        return sum(get_input_characters(input_) for input_ in inputs.values())
    return 0
