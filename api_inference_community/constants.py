from api_inference_community.validation import (
    ConversationalInputsCheck,
    FillMaskParamsCheck,
    QuestionInputsCheck,
    SentenceSimilarityInputsCheck,
    SharedGenerationParams,
    StringInput,
    StringOrStringBatchInputCheck,
    SummarizationParamsCheck,
    TableQuestionAnsweringInputsCheck,
    TabularDataInputsCheck,
    TextGenerationParamsCheck,
    ZeroShotParamsCheck,
)


AUDIO_INPUTS = {
    "automatic-speech-recognition",
    "audio-to-audio",
    "speech-segmentation",
    "audio-classification",
}
AUDIO_OUTPUTS = {"audio-to-audio", "text-to-speech"}


IMAGE_INPUTS = {
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "image-to-image",
    "object-detection",
    "zero-shot-image-classification",
}
IMAGE_OUTPUTS = {"image-to-image", "text-to-image"}


TEXT_INPUTS = {
    "conversational",
    "feature-extraction",
    "question-answering",
    "sentence-similarity",
    "fill-mask",
    "table-question-answering",
    "tabular-classification",
    "tabular-regression",
    "summarization",
    "text-generation",
    "text2text-generation",
    "text-classification",
    "text-to-image",
    "text-to-speech",
    "token-classification",
    "zero-shot-classification",
}


AUDIO = ["flac", "ogg", "mp3", "wav", "m4a", "aac", "webm"]
IMAGE = ["jpeg", "png", "webp", "tiff", "bmp"]

HF_HEADER_COMPUTE_TIME = "x-compute-time"
HF_HEADER_COMPUTE_TYPE = "x-compute-type"

PARAMS_MAPPING = {
    "conversational": SharedGenerationParams,
    "fill-mask": FillMaskParamsCheck,
    "text2text-generation": TextGenerationParamsCheck,
    "text-generation": TextGenerationParamsCheck,
    "summarization": SummarizationParamsCheck,
    "zero-shot-classification": ZeroShotParamsCheck,
}


INPUTS_MAPPING = {
    "conversational": ConversationalInputsCheck,
    "question-answering": QuestionInputsCheck,
    "feature-extraction": StringOrStringBatchInputCheck,
    "sentence-similarity": SentenceSimilarityInputsCheck,
    "table-question-answering": TableQuestionAnsweringInputsCheck,
    "tabular-classification": TabularDataInputsCheck,
    "tabular-regression": TabularDataInputsCheck,
    "fill-mask": StringInput,
    "summarization": StringInput,
    "text2text-generation": StringInput,
    "text-generation": StringInput,
    "text-classification": StringInput,
    "token-classification": StringInput,
    "translation": StringInput,
    "zero-shot-classification": StringInput,
    "text-to-speech": StringInput,
    "text-to-image": StringInput,
}


BATCH_ENABLED_PIPELINES = ["feature-extraction"]
