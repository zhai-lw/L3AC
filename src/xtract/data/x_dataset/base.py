import pickle
from pathlib import Path
from typing import Callable, Iterator, Optional

import datasets

from ..x_feature import XFeature, XFeatureWithInfo

X_FEATURES_FILENAME = "x_features.pkl"


class XFeatureTransformer:
    def __init__(self, x_features: dict[str, XFeature]):
        self.x_features = x_features

    def transform(self, batch_item: dict):
        decoded_item = {}
        for feature_name, encoded_data in batch_item.items():
            x_obj = self.x_features[feature_name]
            if isinstance(x_obj, XFeatureWithInfo):
                decoded_data = x_obj.decode_batch(encoded_data, base_name=feature_name)
                decoded_item.update(decoded_data)
            else:
                decoded_data = x_obj.decode_batch(encoded_data)
                decoded_item[feature_name] = decoded_data
        return decoded_item


class XDataset(datasets.Dataset):
    def __init__(self, *args, **kwargs):
        super(XDataset, self).__init__(*args, **kwargs)
        self.x_features: dict = {}
        self.have_applied_decoder = False

    def apply_decoder(self):
        if self.have_applied_decoder:
            raise RuntimeError("You have applied decoder!")
        else:
            self.have_applied_decoder = True
            self.set_transform(XFeatureTransformer(self.x_features).transform,
                               columns=list(self.x_features), output_all_columns=True)

    def reset_format(self):
        super(XDataset, self).reset_format()
        self.have_applied_decoder = False

    def filter_by_index(self, filtered_index):
        return self.select(filtered_index)

    @classmethod
    def from_dataset(cls, dataset: datasets.Dataset, x_features: dict) -> 'XDataset':
        def warp(ds) -> XDataset:
            ds.__class__ = cls
            return ds

        dataset = warp(dataset)
        dataset.x_features = x_features
        dataset.have_applied_decoder = False
        dataset.apply_decoder()
        return dataset

    @classmethod
    def from_generator(
            cls,
            generator: Callable[..., Iterator[dict]],
            x_features: dict[str, XFeature] = None,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            gen_kwargs: Optional[dict] = None,
            **kwargs
    ):
        x_features = {} if x_features is None else x_features.copy()
        datasets_features = {}
        for feature_name in list(x_features):
            if isinstance(x_features[feature_name], XFeature):
                datasets_features[feature_name] = x_features[feature_name].feature_type
            else:
                datasets_features[feature_name] = x_features.pop(feature_name)

        datasets_features = datasets.Features(datasets_features)

        def x_generator(**g_kwargs):
            for data in generator(**g_kwargs):
                for feature, x_obj in x_features.items():
                    data[feature] = x_obj.check(data[feature])
                yield data

        dataset = super().from_generator(
            generator=x_generator,
            features=datasets_features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        return cls.from_dataset(dataset, x_features)

    def save_to_disk(self, dataset_path: Path, **kwargs):
        self.reset_format()
        super(XDataset, self).save_to_disk(dataset_path=str(dataset_path), **kwargs)
        with Path(dataset_path).joinpath(X_FEATURES_FILENAME).open("wb") as x_features_file:
            pickle.dump(self.x_features, x_features_file)
        self.apply_decoder()

    @classmethod
    def load_from_disk(cls, dataset_path: str | Path, keep_in_memory=None, **kwargs) \
            -> 'XDataset':
        dataset = super().load_from_disk(dataset_path=str(dataset_path),
                                         keep_in_memory=keep_in_memory, **kwargs)
        with Path(dataset_path).joinpath(X_FEATURES_FILENAME).open("rb") as x_features_file:
            x_features = pickle.load(x_features_file)
        return cls.from_dataset(dataset, x_features)
