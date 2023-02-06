from typing import Any, Dict, List, Optional

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class ZeroShotClassificationPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.taskflow = Taskflow(
            "zero_shot_text_classification",
            task_path=model_id,
            from_hf_hub=True,
            pred_threshold=0.0,  # so that it returns all predictions
        )

    def __call__(
        self, inputs: str, candidate_labels: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`): a string to be classified
            candidate_labels (:obj:`List[str]`): a list of strings that are potential classes for inputs.
        Return:
            A :obj:`list`:. a list of dicts containing the following:
                - "sequence": The string sent as an input
                - "labels": The list of strings for labels that you sent (in order)
                - "scores": a list of floats that correspond the the probability of label, in the same order as labels.
        """
        if candidate_labels is None:
            raise ValueError("'candidate_labels' is a required field")
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(",")
        self.taskflow.set_schema(candidate_labels)
        taskflow_results = self.taskflow(inputs)
        pipeline_results = {}
        labels = []
        scores = []
        for result in taskflow_results[0]["predictions"]:
            labels.append(result["label"])
            scores.append(result["score"])
        pipeline_results["labels"] = labels
        pipeline_results["scores"] = scores
        pipeline_results["sequence"] = taskflow_results[0]["text_a"]
        return pipeline_results
