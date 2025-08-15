import xtract
from .base import BaseMetric
from .classify import Accuracy
from .audio import STOI, PESQ
from .vq import CodebookUsage
from ..network import Network


def metric_builder(name: str, network: Network, sample_rate: int, cpu_num: int, cuda_device: int) -> BaseMetric:
    match name.lower():
        case 'accuracy':
            metric = Accuracy()
        case 'stoi':
            metric = STOI(input_sample_rate=sample_rate)
        case 'pesq':
            metric = PESQ(input_sample_rate=sample_rate, cpu_num=cpu_num)
        case 'codebook_usage':
            metric = CodebookUsage(network.quantizer.vq.codebook_size, cuda_device=cuda_device)
        case _:
            raise NotImplementedError(f"{name} not implemented")

    return metric


class Metrics:
    def __init__(self, network: Network, metric_names: list[str], **metric_config):
        self.metrics = [metric_builder(metric_name, network, **metric_config) for metric_name in metric_names]

    def add_metric(self, metric: BaseMetric):
        self.metrics.append(metric)

    def __getitem__(self, metric_name) -> BaseMetric:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric
        raise KeyError(f"{metric_name} not found")

    def reset(self):
        for scorer in self.metrics:
            scorer.reset()

    def update(self, nn_output, ref_input, ):
        for metric in self.metrics:
            metric.update(nn_output, ref_input)

    def compute_internal_results(self) -> dict:
        return {metric.name: metric.compute_internal_results() for metric in self.metrics}

    def get_results(self, internal_results: dict) -> dict:
        metrics_results = {metric_name: self[metric_name].get_results(res)
                           for metric_name, res in internal_results.items()}
        return metrics_results

    def log_results(self, tlog: xtract.tensor_log.Writer | None, namespace: str) -> dict:
        all_results = {}
        for metric in self.metrics:
            internal_results = metric.compute_internal_results()
            if tlog is not None:
                results = metric.get_results(internal_results)
                all_results[metric.name] = metric.log_results(
                    results, tlog=tlog, namespace=f"{namespace}/{metric.name}")
        return all_results
