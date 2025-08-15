import abc
from collections import OrderedDict
from typing import Iterable

import datasets
import numpy
import lz4.frame


class XFeature(abc.ABC):
    feature_type = datasets.Value('binary')

    def encode(self, data):
        if data is None:
            return b''
        else:
            return self.encode_data(data)

    def decode(self, x_data):
        if x_data == b'':
            return None
        elif x_data is None:
            return None
        else:
            return self.decode_data(x_data)

    def check(self, data):
        if data is not None:
            encoded_feature = self.encode(data)
            assert self.is_equal(self.decode(encoded_feature), data)
            return encoded_feature

    def decode_batch(self, x_batch: list) -> list:
        return [self.decode(x_data) for x_data in x_batch]

    def encode_data(self, data):
        raise NotImplementedError

    def decode_data(self, x_data):
        raise NotImplementedError

    @staticmethod
    def is_equal(decoded_data, original_data) -> bool:
        raise NotImplementedError


class XFeatureWithInfo(XFeature):

    def decode_data(self, x_data) -> dict:
        raise NotImplementedError

    def decode_batch(self, x_batch: list, base_name='{}') -> dict:
        decode_list = super().decode_batch(x_batch)
        return {
            name.format(base_name): [decode_data[name] for decode_data in decode_list]
            for name in decode_list[0].keys()
        }


class XLabel(XFeatureWithInfo):
    feature_type = datasets.Value('uint32')

    def __init__(self, name_info: OrderedDict = None):
        from bidict import bidict
        if name_info is None:
            self.label_name = bidict()
            self.label_info = self.label_name
        else:
            self.label_name = bidict(enumerate(name_info))
            self.label_info = {label: (name, name_info[name]) for label, name in self.label_name.items()}

    @property
    def name_label(self):
        return self.label_name.inverse

    @property
    def editable(self):
        return self.label_name == self.label_info

    def _put_name(self, name):
        if not self.editable:
            raise RuntimeError("Cannot update a initialized XLabel instance")
        label = len(self.label_name)
        if label.bit_length() < 32:
            raise RuntimeError(f"Label overflow: {label} ({label.bit_length()}bits)")
        self.label_name.put(label, name)
        return label

    def encode_data(self, name) -> int:
        if name not in self.name_label:
            return self._put_name(name)
        else:
            return self.name_label[name]

    def decode_data(self, label: int) -> dict:
        return {'{}': label, '{}_info': self.label_info[label]}

    @staticmethod
    def is_equal(decoded_data, original_data) -> bool:
        name = decoded_data['{}_info']
        name = name[0] if isinstance(name, tuple) else name
        return name == original_data


class XBytes(XFeature):

    def __init__(self, compression_level: int = 5):
        self.compression_level: int = compression_level

    def encode_data(self, data: bytes) -> bytes:
        # x_data = pyarrow.compress(value, 'lz4', asbytes=True)
        x_data = lz4.frame.compress(data, compression_level=self.compression_level)
        return x_data

    def decode_data(self, x_data: bytes) -> bytes:
        data = lz4.frame.decompress(x_data)
        return data

    @staticmethod
    def is_equal(decoded_data, original_data) -> bool:
        check_result = (decoded_data == original_data)
        if isinstance(check_result, Iterable):
            return all(check_result)
        else:
            return check_result


class XArray(XBytes):
    def __init__(self, shape: tuple, dtype: str, compression_level: int = 5):
        super().__init__(compression_level=compression_level)
        self.shape = tuple(shape)
        self.dtype = dtype

    def encode_data(self, data: numpy.ndarray) -> bytes:
        compressed_data = super().encode_data(data.tobytes())
        return compressed_data

    def decode_data(self, x_data: bytes) -> numpy.ndarray:
        decompressed_data = super().decode_data(x_data)
        data = numpy.frombuffer(decompressed_data, dtype=self.dtype)
        return data.reshape(self.shape)

    @staticmethod
    def is_equal(decoded_data, original_data) -> bool:
        return numpy.array_equal(decoded_data, original_data)  # set equal_nan=False to check nan
