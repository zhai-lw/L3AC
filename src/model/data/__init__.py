import random
from functools import cached_property, partial
from pathlib import Path

import numpy
from pydantic import BaseModel, Field, field_validator, ValidationError, ValidationInfo
import torch.utils.data

from utils.file import PROJECT_PATH
from xtract.data import XDataset, x_dataset

from prepare.data_process import DN
from .format import DataCollector

DEFAULT_DATASET_DIR = (PROJECT_PATH / 'data/dataset')


class DataLoaderBuilder(BaseModel):
    dataset_paths: list[Path]

    sample_rate: int = 16000
    data_normalize: bool = True
    max_seconds: float = 5.9

    batch_size: int = 4
    sample_num: list[int | None] = Field(default=None, validate_default=True)

    num_workers: int = 1

    @field_validator('sample_num', mode='before')
    @classmethod
    def sample_num_validator(cls, sample_num: list | int | None, info: ValidationInfo):
        num_datasets = len(info.data['dataset_paths'])
        if isinstance(sample_num, int):
            sample_num = [sample_num // num_datasets, ] * num_datasets
        elif sample_num is None:
            sample_num = [None] * num_datasets
        if len(sample_num) != num_datasets:
            raise ValidationError(f"Got sample_num({sample_num}) but {num_datasets=}")
        return sample_num

    @cached_property
    def datasets(self) -> list[DataCollector]:
        all_datasets = []
        for ds_path in self.dataset_paths:
            if not ds_path.is_absolute():
                ds_path = DEFAULT_DATASET_DIR / ds_path
            dataset = XDataset.load_from_disk(ds_path)
            ds_name = ".".join(ds_path.parts[-2:])
            dataset = DataCollector(dataset, ds_name=ds_name, sample_rate=self.sample_rate,
                                    data_normalize=self.data_normalize)
            all_datasets.append(dataset)
        return all_datasets

    def get_dataloader(self, prefetch_size: int = None) -> torch.utils.data.DataLoader:
        all_datasets = []
        for dataset, sample_num in zip(self.datasets, self.sample_num):
            if sample_num is not None:
                selected_indices = numpy.random.permutation(len(dataset))[:sample_num]
                dataset = torch.utils.data.Subset(dataset, selected_indices.tolist())
            all_datasets.append(dataset)
        all_dataset = torch.utils.data.ConcatDataset(all_datasets)

        collate_fn = partial(collate_data_items,
                             max_data_len=int(self.max_seconds * self.sample_rate),
                             collect_key_default={
                                 DN.audio: None,
                                 DN.id: '',
                                 DN.name: '',
                                 DN.txt: '',
                             })
        dataloader = x_dataset.init_dataloader(
            all_dataset, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_size=prefetch_size,
            collate_fn=collate_fn, shuffle=True, pin_memory=True, drop_last=True,
        )
        return dataloader


def pad_or_clip(audio_data: numpy.ndarray, target_length: int):
    if audio_data.size > target_length:
        start_index = random.randint(0, audio_data.size - target_length)
        audio_data = audio_data[start_index:start_index + target_length]
    else:
        # audio_data = numpy.pad(audio_data, (target_length - audio_data.size, 0))  # ! left pad or right pad?
        audio_data = numpy.pad(audio_data, (0, target_length - audio_data.size))
    return audio_data


def collate_data_items(data_items: list[dict], max_data_len: int,
                       collect_key_default: dict):
    data_items = {k: [data_items[i].get(k, default_v) for i in range(len(data_items))]
                  for (k, default_v) in collect_key_default.items()}
    max_data_length = max(map(len, data_items[DN.audio]))
    target_data_length = min(max_data_length, max_data_len)
    data_items[DN.audio] = [pad_or_clip(audio_data, target_length=target_data_length)
                            for audio_data in data_items[DN.audio]]
    return {key: torch.utils.data.default_collate(item) for key, item in data_items.items()}
