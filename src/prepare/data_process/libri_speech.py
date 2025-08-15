# pip install 'datasets<4.0'
import datasets

import runtime_resolver
import utils
from xtract.data import x_dataset, x_feature

from prepare.data_process import DN

RS = runtime_resolver.init_runtime()

log = utils.log.get_logger()

DATASET_SAMPLE_RATE = 16000
DATASET_CHANNELS = 1
AUDIO_FMT = 'FLAC'

DATA_DIR = RS.data_path / "dataset"

(TARGET_DIR := DATA_DIR / f"libri_speech_{AUDIO_FMT}").mkdir(exist_ok=True)

x_features = {
    DN.name: x_feature.Value('string'),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
    DN.txt: x_feature.Value('string'),
}


def init_dataset(src_dataset: datasets.Dataset) -> x_dataset.XDataset:
    def iter_data(indexes):
        for data_idx in indexes:
            data_item = src_dataset[data_idx]
            _, audio_data, sample_rate = data_item['audio'].values()
            assert sample_rate == DATASET_SAMPLE_RATE
            assert len(audio_data.shape) == 1
            yield {
                DN.name: data_item['id'],
                DN.audio: audio_data,
                DN.txt: data_item['text'],
            }

    # for item in iter_data(range(10)):
    #     print(item)

    dataset = x_dataset.XDataset.from_generator(iter_data,
                                                gen_kwargs=dict(indexes=list(range(len(src_dataset)))),
                                                x_features=x_features, num_proc=RS.cpu_num)
    return dataset


def init():
    log.info(f"start init dataset")
    import aiohttp
    src_datadict = datasets.load_dataset("librispeech_asr", "clean",
                                         trust_remote_code=True,
                                         storage_options={
                                             'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=36000)}})

    for ds_name, src_dataset in {
        "test": src_datadict['test'],
        "eval": src_datadict['validation'],
        "train": datasets.concatenate_datasets([src_datadict['train.100'], src_datadict['train.360']]),
    }.items():
        log.info(f"init {ds_name} dataset ")
        dataset = init_dataset(src_dataset)
        dataset.save_to_disk(TARGET_DIR / ds_name)


if __name__ == '__main__':
    init()
