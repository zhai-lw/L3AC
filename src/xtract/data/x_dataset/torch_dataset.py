import math
import datasets
import torch.utils.data


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset, data_id_formatter=None, ds_name: str = "ds"):
        self.dataset = dataset

        if data_id_formatter is None:
            data_id_max_len = len(str(len(self.dataset)))
            self.data_id_formatter = f'{ds_name}.{{0:0{data_id_max_len}d}}'
        else:
            self.data_id_formatter = data_id_formatter

    def __getitem__(self, index):
        raise RuntimeError("should never be called")

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, keys):
        items = self.dataset[keys]
        items['data_id'] = [self.data_id_formatter.format(key) for key in keys]
        return items


def default_collate_fn(items: list | dict):
    return {name: torch.utils.data.default_collate(item) for name, item in items.items()}


def init_dataloader(dataset: torch.utils.data.Dataset,
                    batch_size: int, num_workers: int, prefetch_size: int = None,
                    collate_fn=default_collate_fn, shuffle=True, pin_memory=True, drop_last=True,
                    **kwargs) -> torch.utils.data.DataLoader:
    if prefetch_size is None:
        prefetch_factor = 2
    else:
        prefetch_factor = math.ceil(float(prefetch_size) / batch_size / num_workers)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, collate_fn=collate_fn,
                                       pin_memory=pin_memory, drop_last=drop_last,
                                       prefetch_factor=prefetch_factor,
                                       **kwargs, )
