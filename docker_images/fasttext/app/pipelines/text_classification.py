from typing import Dict, List

from app.pipelines import Pipeline
from huggingface_hub import HfApi


FASTTEXT_PREFIX_LENGTH = 9  # fasttext labels are formatted like "__label__eng_Latn"


class TextClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        super().__init__(model_id)
        self.info = HfApi().model_info(repo_id=self.model_id)

    def __call__(self, inputs: str) -> List[Dict[str, float]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing:
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """

        if "language-identification" in self.info.tags:
            preds = self.model.predict(inputs, k=5)
            result = [
                {"label": label[FASTTEXT_PREFIX_LENGTH:], "score": prob}
                for label, prob in zip(preds[0], preds[1])
            ]
            return [result]

        if len(inputs.split()) > 1:
            raise ValueError("Expected input is a single word")
        preds = self.model.get_nearest_neighbors(inputs, k=5)
        result = []
        for distance, word in preds:
            result.append({"label": word, "score": distance})
        return [result]
