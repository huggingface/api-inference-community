from app.pipelines.base import Pipeline, PipelineException  # isort:skip

from app.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from app.pipelines.text_classification import TextClassificationPipeline
from app.pipelines.token_classification import TokenClassificationPipeline
from app.pipelines.text2text_generation import TextToTextPipeline
from app.pipelines.fill_mask import FillMaskPipeline
from app.pipelines.image_to_text import ImageToTextPipeline
