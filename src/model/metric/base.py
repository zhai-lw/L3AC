from collections import defaultdict

import xtract
from xtract import ddp


class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    def reset(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute_internal_results(self):
        raise NotImplementedError

    @classmethod
    def get_results(cls, internal_results) -> dict:
        raise NotImplementedError

    @classmethod
    def log_results(cls, results: dict, tlog: xtract.tensor_log.Writer, namespace: str) -> dict:
        for name, result in results.items():
            tlog.variable(name=f"{namespace}/{name}", value=result, prog_bar=True)
        return results


class ScoreMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.scores_dict = defaultdict(list)
        self.scores = self.scores_dict[self.name]

    def reset(self):
        for sl in self.scores_dict.values():
            sl.clear()

    def update(self, *args, **kwargs):
        raise NotImplementedError

    @ddp.dict_list_gather
    def compute_internal_results(self) -> dict:
        return self.scores_dict

    @classmethod
    def get_results(cls, scores_dict: dict) -> dict:
        return {score_name: (sum(scores) / len(scores) if len(scores) > 0 else -1.0)
                for score_name, scores in scores_dict.items()}
