from typing import Dict, List

from app.pipelines.common import SklearnBasePipeline


class TextClassificationPipeline(SklearnBasePipeline):
    def _get_output(self, inputs: str) -> List[Dict[str, float]]:
        res = []
        for i, c in enumerate(self.model.predict_proba([inputs]).tolist()[0]):
            res.append({"label": str(self.model.classes_[i]), "score": c})
        return [res]
