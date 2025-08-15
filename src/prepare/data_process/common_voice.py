# download common voice dataset from https://commonvoice.mozilla.org
# extract cv-corpus-18.0-2024-06-14-en.tar.gz to data/dataset/source/common_voice/cv-corpus-18.0-2024-06-14/

import pandas

import runtime_resolver
import tools.audio
import utils
from xtract.data import x_dataset, x_feature

from prepare.data_process import DN

RS = runtime_resolver.init_runtime()

log = utils.log.get_logger()

DATASET_SAMPLE_RATE = 24000
DATASET_CHANNELS = 1
AUDIO_FMT = 'MP3-high'

DATA_DIR = RS.data_path / "dataset"

SOURCE_DIR = DATA_DIR / "source" / "common_voice/cv-corpus-18.0-2024-06-14/en/"
(TARGET_DIR := DATA_DIR / "common_voice_24k").mkdir(exist_ok=True)

x_features = {
    DN.name: x_feature.Value('string'),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
    DN.txt: x_feature.Value('string'),
}


def init_dataset(index_data: pandas.DataFrame) -> x_dataset.XDataset:
    load_audio = tools.audio.load

    def iter_data(indexes):
        for data_idx in indexes:
            data_item = index_data.loc[data_idx]
            audio_path = data_item['path']
            audio_data = load_audio(f"{SOURCE_DIR}/clips/{audio_path}",
                                    channels=DATASET_CHANNELS, frame_rate=DATASET_SAMPLE_RATE)
            yield {
                DN.name: audio_path,
                DN.audio: audio_data,
                DN.txt: data_item['sentence'],
                # 'up_votes': data_item['up_votes'],
                # 'down_votes': data_item['down_votes'],
            }

    # for item in iter_data(range(10)):
    #     print(item)

    dataset = x_dataset.XDataset.from_generator(iter_data,
                                                gen_kwargs=dict(indexes=list(index_data.index)),
                                                x_features=x_features, num_proc=RS.cpu_num)
    return dataset


def init():
    log.info(f"start init dataset")

    for ds_name, index_data in {
        "test": pandas.read_csv(SOURCE_DIR / f"test.tsv", sep='\t'),
        "eval": pandas.read_csv(SOURCE_DIR / f"dev.tsv", sep='\t'),
        "train": pandas.read_csv(SOURCE_DIR / f"train.tsv", sep='\t'),
    }.items():
        log.info(f"init {ds_name} dataset ")
        dataset = init_dataset(index_data)
        dataset.save_to_disk(TARGET_DIR / ds_name)


if __name__ == '__main__':
    init()
