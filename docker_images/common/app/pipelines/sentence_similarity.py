from typing import List

from app.pipelines import Pipeline


class SentenceSimilarityPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement SentenceSimilarityPipeline __init__ function"
        )

    def __call__(self, source_sentence: str, sentences: List[str]) -> List[float]:
        """
        Args:
            source_sentence (:obj:`str`):
                a string that will be compared to every sentence in `sentences`.
            sentences (:obj:`List[str]`):
                a list of sentences to be compared against source_sentence.
        Return:
            A :obj:`list` of floats: Some similarity measure between `source_sentence` and each sentence from `sentences`.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement SentenceSimilarityPipeline __call__ function"
        )
