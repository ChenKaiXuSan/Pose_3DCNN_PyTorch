'''
A pytorch lightning data module based dataloader, for train/val/test dataset prepare.
rewrite the WalkLabeledVideoDataset class, where from pytorchvideo.
make sure use same torch.Generator() to get the same order when load four fidderent dataset.

Note that, based pose method, we can't use more data augument technology such as rotation, translation, clipping.
Because of we need to ensure the correct locate of different part for splitted part.

'''

# %%
import matplotlib.pylab as plt

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Div255,
    create_video_transform,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
import os

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset, labeled_video_dataset, LabeledVideoPaths
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# different part of person
PART = ['body', 'head', 'upper', 'lower']

# %%
# make sure different dataset use same generator.
VIDEO_RANDOM_GENERATOR = torch.Generator()

class WalkLabeledVideoDataset(LabeledVideoDataset):
    def __init__(self, labeled_video_paths: List[Tuple[str, Optional[dict]]], clip_sampler: ClipSampler, video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler, transform: Optional[Callable[[dict], Any]] = None, decode_audio: bool = True, decoder: str = "pyav") -> None:
        super().__init__(labeled_video_paths, clip_sampler, video_sampler, transform, decode_audio, decoder)
        
        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        # self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=VIDEO_RANDOM_GENERATOR, 
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)


def WalkDataset(
    data_path: str,
    flag: str,
    part: str,
    clip_duration: int,
    # clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> list:
    '''
    A helper function to create "LabeledVideoDataset" object for the Walk dataset.

    Args:
        data_path (str): Path to the data. The path defines how the data should be read. For a directory, the directory structure defines the classes (i.e. each subdirectory is class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler], optional): Sampler for the internal video container. Defaults to torch.utils.data.RandomSampler.
        transform (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], optional): This callable is evaluated on the clip output before the clip is returned. Defaults to None.
        video_path_prefix (str, optional): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. Defaults to "".
        decode_audio (bool, optional): If True, also decode audio from video. Defaults to False. Notice that, if Ture will trigger the stack error.
        decoder (str, optional): Defines what type of decoder used to decode a video. Defaults to "pyav".

    Returns:
        LabeledVideoDataset: _description_
    '''

    labeled_video_list = []

    if part == 'all':
        for part in PART:
            PATH = os.path.join(data_path, part, flag)

            print("#" * 50)
            print("load path:", PATH)
            print("#" * 50)

            labeled_video_paths = LabeledVideoPaths.from_path(PATH)
            labeled_video_paths.path_prefix = video_path_prefix
            dataset = WalkLabeledVideoDataset(
                    labeled_video_paths,
                    make_clip_sampler("uniform", clip_duration),
                    video_sampler,
                    transform,
                    decode_audio=decode_audio,
                    decoder=decoder,
                )

            labeled_video_list.append(
                dataset
            )

    else:

        PATH = os.path.join(data_path, part, flag)

        print("#" * 50)
        print("load path:", PATH)
        print("#" * 50)

        labeled_video_list.append(
            labeled_video_dataset(
                PATH,
                make_clip_sampler('random', clip_duration),
                video_sampler,
                transform=transform,
                video_path_prefix=video_path_prefix,
                decode_audio=decode_audio,
                decoder=decoder
            )
        )

    return labeled_video_list

# %%


class WalkDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        # use this for dataloader
        self._TRAIN_PATH = opt.train_path

        self._PRE_PROCESS_FLAG = opt.pre_process_flag

        self._BATCH_SIZE = opt.batch_size
        self._NUM_WORKERS = opt.num_workers
        self._IMG_SIZE = opt.img_size

        self._PART = opt.part

        # frame rate
        self._CLIP_DURATION = opt.clip_duration
        self.uniform_temporal_subsample_num = opt.uniform_temporal_subsample_num

        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),

                            # dived the pixel from [0, 255] tp [0, 1], to save computing resources.
                            Div255(),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            # RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(self._IMG_SIZE),

                            # ShortSideScale(self._IMG_SIZE),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            # RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

        self.raw_train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),

                            # dived the pixel from [0, 255] to [0, 1], to save computing resources.
                            Div255(),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            RandomShortSideScale(min_size=256, max_size=320),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    )
                )
            ]
        )

    def prepare_data(self) -> None:
        # used for download dataset
        # if self._PRE_PROCESS_FLAG:
        #     data_path = self._TRAIN_PATH
        #     transform = self.train_transform
        #     print("#" * 50)
        #     print("run pre process model!")
        #     print("#" * 50)

        # else:
        #     data_path = self._TRAIN_PATH
        #     transform = self.raw_train_transform
        #     print("#" * 50)
        #     print("run not pre process model!")
        #     print("#" * 50)

        # self.train_dataset = WalkDataset(
        #         data_path=data_path, flag='train', part=self._PART,
        #         clip_duration=self._CLIP_DURATION,
        #         transform=transform,
        #     )

        # self.val_dataset = WalkDataset(
        #     data_path=data_path, flag='val', part=self._PART,
        #     clip_duration=self._CLIP_DURATION,
        #     transform=transform,
        # )
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        '''
        if self._PRE_PROCESS_FLAG:
            data_path = self._TRAIN_PATH
            transform = self.train_transform
            print("#" * 50)
            print("run pre process model!")
            print("#" * 50)

        else:
            data_path = self._TRAIN_PATH
            transform = self.raw_train_transform
            print("#" * 50)
            print("run not pre process model!")
            print("#" * 50)

        # if stage == "fit" or stage == None:
        if stage in ("fit", None):
            self.train_dataset = WalkDataset(
                data_path=data_path, flag='train', part=self._PART,
                clip_duration=self._CLIP_DURATION,
                transform=transform,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = WalkDataset(
                data_path=data_path, flag='val', part=self._PART,
                clip_duration=self._CLIP_DURATION,
                transform=transform,
            )

        if stage in ("predict", "test", None):
            self.test_pred_dataset = WalkDataset(
                data_path=data_path, flag='val', part=self._PART,
                clip_duration=self._CLIP_DURATION,
                transform=transform
            )

    def train_dataloader(self) -> dict:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''

        pose_Dict = {}

        if self._PART != 'all':
            _PART = [self._PART]
        else:
            _PART = PART

        for i, p in enumerate(_PART):
            pose_Dict[p] = DataLoader(
                self.train_dataset[i],
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

        combined_loaders = CombinedLoader(pose_Dict, mode='min_size')

        return combined_loaders

    def val_dataloader(self) -> dict:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''

        pose_Dict = {}

        if self._PART != 'all':
            _PART = [self._PART]
        else:
            _PART = PART

        for i, p in enumerate(_PART):
            pose_Dict[p] = DataLoader(
                self.val_dataset[i],
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                drop_last=True
            )

        combined_loaders = CombinedLoader(pose_Dict, mode='min_size')

        return combined_loaders

    def test_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and 
        normalizes the video before applying the scale, crop and flip augmentations.
        '''
        return DataLoader(
            self.test_pred_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def predict_dataloader(self) -> DataLoader:
        '''
        create the Walk pred partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''
        return DataLoader(
            self.test_pred_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
