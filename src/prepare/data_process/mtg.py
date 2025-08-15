# git clone https://github.com/MTG/mtg-jamendo-dataset.git
# mkdir mtg-jamendo-dataset/data/audios
# download the audio tar files to mtg-jamendo-dataset/data/audios
# link mtg-jamendo-dataset/data to data/dataset/source/mtg-jamendo-dataset

import pathlib
import tarfile

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
AUDIO_FMT = 'MP3-medium'

DATA_DIR = RS.data_path / "dataset"

SOURCE_DIR = DATA_DIR / "source" / "mtg-jamendo-dataset"
(TARGET_DIR := DATA_DIR / "mtg_low").mkdir(exist_ok=True)

x_features = {
    DN.name: x_feature.Value('string'),
    DN.audio: x_feature.extension.XWave(compress_fmt=AUDIO_FMT, frame_rate=DATASET_SAMPLE_RATE),
    'clip_idx': x_feature.Value('uint32'),
    'audio_duration': x_feature.Value('float32'),
}

audio_load_func = tools.audio.load


class MTGDataBase:
    def __init__(self):
        self.audio_dir = SOURCE_DIR / "audios"
        self.tar_files = {}

    def get_tar_file(self, dir_name):
        if dir_name not in self.tar_files:
            tar_file = self.audio_dir / f"raw_30s_audio-low-{dir_name}.tar"
            self.tar_files[dir_name] = tarfile.open(tar_file, 'r')
        return self.tar_files[dir_name]

    def get_audio_data(self, audio_name):
        dir_name, audio_name = audio_name.split('/')
        tar_file = self.get_tar_file(dir_name)
        f = tar_file.extractfile(f"{dir_name}/{audio_name.removesuffix('.mp3')}.low.mp3")
        audio_data = audio_load_func(f, channels=DATASET_CHANNELS, frame_rate=DATASET_SAMPLE_RATE)
        return audio_data


def iter_metadata(file_path: pathlib.Path):
    with file_path.open('r') as reader:
        tags_start_idx = 5
        assert 'TAGS' == reader.readline().rstrip().split('\t')[tags_start_idx]
        for i, line in enumerate(reader.readlines()):
            ls = line.rstrip().split('\t')
            track_id, artist_id, album_id, audio_name, duration = ls[:tags_start_idx]

            track_data = {
                DN.name: audio_name,
                # 'track_id': int(track_id.removeprefix('track_')),
                # 'album_id': int(album_id.removeprefix('album_')),
                # 'artist_id': int(artist_id.removeprefix('artist_')),
                'audio_duration': float(duration),
                # **format_tags(ls[tags_start_idx:])
            }

            yield track_data


def format_tags(tags):
    genres, instruments, moods = [], [], []
    for tag in tags:
        match tag.split('---'):
            case 'genre', tag_name:
                genres.append(tag_name)
            case 'instrument', tag_name:
                instruments.append(tag_name)
            case 'mood/theme', tag_name:
                moods.append(tag_name)
            case _:
                raise ValueError(f"unknown tag {tag}")

    return {"genres": genres, "instruments": instruments, "moods": moods}


def init_dataset(meta_data_file: pathlib.Path):
    meta_data = pandas.DataFrame(iter_metadata(meta_data_file))
    db = MTGDataBase()
    clip_length = 25 * DATASET_SAMPLE_RATE

    def iter_data(indexes):
        for data_idx in indexes:
            data_item = meta_data.loc[data_idx]
            audio_data = db.get_audio_data(data_item[DN.name])
            for idx in range(0, len(audio_data), clip_length):
                yield {
                    DN.name: data_item[DN.name],
                    DN.audio: audio_data[idx:idx + clip_length],
                    'clip_idx': idx,
                    'audio_duration': data_item['audio_duration'].item(),
                }

    # for item in iter_data(range(10)):
    #     print(item)

    dataset = x_dataset.XDataset.from_generator(iter_data,
                                                gen_kwargs=dict(indexes=list(meta_data.index)),
                                                x_features=x_features, num_proc=RS.cpu_num)
    return dataset


def init():
    log.info(f"start init dataset")

    meta_data_dir = SOURCE_DIR / "splits/split-0"
    for ds_name, split_name in dict(train="train", eval="validation", test="test").items():
        log.info(f"init {ds_name} dataset ")
        meta_data_file = meta_data_dir / f"autotagging-{split_name}.tsv"
        dataset = init_dataset(meta_data_file)
        dataset.save_to_disk(TARGET_DIR / ds_name)


if __name__ == '__main__':
    init()
