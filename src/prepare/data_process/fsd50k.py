# download the data and extract to data/dataset/source/FSD50K/(FSD50K.dev_audio,FSD50K.eval_audio,FSD50K.metadata)

import pathlib
import pandas

import runtime_resolver
import tools.audio
import utils
from xtract.data import x_dataset, x_feature

from prepare.data_process import DN

RS = runtime_resolver.init_runtime()

log = utils.log.get_logger()

DATASET_SAMPLE_RATE = 44100
DATASET_CHANNELS = 1
AUDIO_FMT = 'FLAC'

DATA_DIR = RS.data_path / "dataset"

SOURCE_DIR = DATA_DIR / "source" / "FSD50K"
(TARGET_DIR := DATA_DIR / "FSD50K_44k").mkdir(exist_ok=True)

x_features = {
    DN.name: x_feature.Value('string'),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
}


def init_dataset(meta_data_path: pathlib.Path, audio_dir: pathlib.Path):
    load_audio = tools.audio.load

    import json
    with meta_data_path.open("r") as meta_data_file:
        meta_data_dict = json.load(meta_data_file)

    meta_data = pandas.DataFrame.from_dict(meta_data_dict, orient="index")

    def iter_data(indexes):
        for data_idx in indexes:
            # data_item = meta_data.loc[data_idx]
            audio_data = load_audio(audio_dir / f"{data_idx}.wav",
                                    channels=DATASET_CHANNELS, frame_rate=DATASET_SAMPLE_RATE)
            yield {
                DN.name: data_idx,
                DN.audio: audio_data,
                # "data_description": data_item["description"],
                # "data_tags": data_item["tags"],
            }

    # for item in iter_data(['100', '1001', '1005']):
    #     print(item)

    dataset = x_dataset.XDataset.from_generator(iter_data,
                                                gen_kwargs=dict(indexes=list(meta_data.index)),
                                                x_features=x_features, num_proc=RS.cpu_num)
    return dataset


def init():
    log.info(f"start init dataset")

    for ds_name, split_name in dict(eval="eval", train="dev", ).items():
        log.info(f"init {ds_name} dataset ")
        meta_data_file = SOURCE_DIR / "FSD50K.metadata" / f"{split_name}_clips_info_FSD50K.json"
        dataset = init_dataset(meta_data_file, SOURCE_DIR / f"FSD50K.{split_name}_audio")
        dataset.save_to_disk(TARGET_DIR / ds_name)


if __name__ == '__main__':
    init()
