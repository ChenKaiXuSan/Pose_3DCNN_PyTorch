'''
batch split method.
this file define how to process the one batch video into different part.
aspire one batch video.
'''
# %%
import warnings

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from torchvision.io import read_image, read_video, write_video
from torchvision.utils import save_image
from torchvision.transforms.functional import crop, resize, pad

# %%

warnings.filterwarnings('ignore')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# %%
class BatchSplit():

    def __init__(self, img_size=224, uniform_temporal_subsample_num=10) -> None:


        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                 model_complexity=2, enable_segmentation=True)

        self.img_size = img_size
        self.unifrom_temporal_subsample_num = uniform_temporal_subsample_num

    def mask_pose(self,
                  image: torch.Tensor):

        # c, h, w to h, w, c
        annotated_image = image.cpu().numpy().copy()

        results = self.pose.process(annotated_image.astype(np.uint8))

        if results.segmentation_mask is not None and results.pose_landmarks is not None:
            # draw mask to img
            segm_2class = results.segmentation_mask
            segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
            masked_img = annotated_image * segm_2class

            # post estimation
            pose = results.pose_landmarks.landmark

            return torch.from_numpy(masked_img), pose
        else:
            return None, None

    def get_center_point(
        self,
        left_keypoint,
        right_keypoint
    ):
        x = abs(left_keypoint[0] - right_keypoint[0]) / 2
        y = abs(left_keypoint[1] - right_keypoint[1]) / 2

        if left_keypoint[0] < right_keypoint[0]:
            X = left_keypoint[0] + x
        else:
            X = right_keypoint[0] + x

        if left_keypoint[1] < right_keypoint[1]:
            Y = left_keypoint[1] + y
        else:
            Y = left_keypoint[1] + y

        return (X, Y)

    def split_head_part(self,
                        image: torch.Tensor,
                        keypoint,
                        bias = 50
                        ):

        width, height, c = image.shape

        # split the image
        ear_center_point = self.get_center_point(
            (keypoint[mp_pose.PoseLandmark.LEFT_EAR].x,
             keypoint[mp_pose.PoseLandmark.LEFT_EAR].y),
            (keypoint[mp_pose.PoseLandmark.RIGHT_EAR].x,
             keypoint[mp_pose.PoseLandmark.RIGHT_EAR].y)
        )

        shoulder_center_point = self.get_center_point(
            (
                keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            ),
            (
                keypoint[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                keypoint[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            )
        )

        head_img = crop(image.permute(2, 0, 1),
                        top=0, left=int(
                        ear_center_point[0] * width - bias),
                        height=int(shoulder_center_point[1] * height), width=int(shoulder_center_point[1] * height))
        head_img = resize(head_img, [self.img_size, self.img_size])

        return head_img

    def split_upper_part(
        self,
        image: torch.Tensor,
        keypoint,
        bias=60
    ):

        width, height, c = image.shape

        shoulder_center_point = self.get_center_point(
            (keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
             keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (keypoint[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
             keypoint[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        )

        hip_center_point = self.get_center_point(
            (keypoint[mp_pose.PoseLandmark.LEFT_HIP].x,
             keypoint[mp_pose.PoseLandmark.LEFT_HIP].y),
            (keypoint[mp_pose.PoseLandmark.RIGHT_HIP].x,
             keypoint[mp_pose.PoseLandmark.RIGHT_HIP].y)
        )
        body_height = abs(hip_center_point[1] - shoulder_center_point[1]) * height

        if body_height > 1:

            body_img = crop(image.permute(2, 0, 1),
                            top=int(shoulder_center_point[1] * height),
                            left=int(shoulder_center_point[0] * width - bias),
                            height=int(body_height+bias), width=int(body_height+bias))

            # pad_body_img = pad(body_img, padding=(0, int(bias/2)), fill=0)

            body_img = resize(body_img, [self.img_size, self.img_size])

            save_image(body_img/255, fp='body.png')

            return body_img

    def split_lower_part(
        self, 
        image: torch.Tensor,
        keypoint
    ):

        width, height, c = image.shape

        hip_center_point = self.get_center_point(
            (keypoint[mp_pose.PoseLandmark.LEFT_HIP].x,
             keypoint[mp_pose.PoseLandmark.LEFT_HIP].y),
            (keypoint[mp_pose.PoseLandmark.RIGHT_HIP].x,
             keypoint[mp_pose.PoseLandmark.RIGHT_HIP].y)
        )

        lower_height = abs(1 - hip_center_point[1]) * height

        if lower_height > 1:
                
            lower_img = crop(image.permute(2, 0, 1),
                            top=int(hip_center_point[1] * height),
                            left=int(hip_center_point[0] * width - 60),
                            height=int(lower_height), width=int(lower_height))
            lower_img = resize(lower_img, [self.img_size, self.img_size])

            return lower_img

    def check_list(self, index):
        '''
        check list, to make the size be same

        Args:
            index (int): need operate  index, used in different list.
        '''        
        
        self.masked_list.pop(index)
        self.head_list.pop(index)
        self.upper_list.pop(index)
        self.lower_list.pop(index)
        
        # avoid out of list err.
        if len(self.masked_list) <= index:
            index -= 1

        self.masked_list.append(self.masked_list[index])
        self.head_list.append(self.head_list[index])
        self.upper_list.append(self.upper_list[index])
        self.lower_list.append(self.lower_list[index])

    def check_T(self):
        '''
        check T dimension, to make sure differnt batch have same T.
        if not, copy the last frame.
        '''        
        while True:
            if len(self.masked_list) != self.unifrom_temporal_subsample_num \
                and len(self.head_list) != self.unifrom_temporal_subsample_num \
                and len(self.upper_list) != self.unifrom_temporal_subsample_num \
                and len(self.lower_list) != self.unifrom_temporal_subsample_num:

                self.masked_list.append(self.masked_list[-1])
                self.head_list.append(self.head_list[-1])
                self.upper_list.append(self.upper_list[-1])
                self.lower_list.append(self.lower_list[-1])
            else:
                break


    def handle_batch_video(self, video):
        
        T, H, W, C = video.size()
            
        #################
        # start one batch 
        #################

        batch_video = video
        
        # init list for every batch
        self.keypoints_list = []
        self.masked_list = []
        self.head_list = []
        self.upper_list = []
        self.lower_list = []

        for i in range(T):

            masked_image, keypoints = self.mask_pose(batch_video[i])

            if masked_image is not None and keypoints is not None:
                self.keypoints_list.append(keypoints)  # 1, 17, 3
                self.masked_list.append(masked_image)

        # check masked list not empty.
        if len(self.masked_list) != 0:

            for m, k in zip(self.masked_list, self.keypoints_list):
                self.head_list.append(self.split_head_part(m, k))
                self.upper_list.append(self.split_upper_part(m, k))
                self.lower_list.append(self.split_lower_part(m, k))

            # when not kpt pred but masked list have some frame, not drop batch but copy the last frame.
            while True:
                if None in self.head_list:
                    index = self.head_list.index(None)

                    self.check_list(index)

                elif None in self.upper_list:
                    index = self.upper_list.index(None)

                    self.check_list(index)

                elif None in self.lower_list:
                    index = self.lower_list.index(None)

                    self.check_list(index)

                else: 
                    # check if lost frame
                    # self.check_T()
                    break;

        # list check not empty.
        if len(self.head_list) != 0 \
            and len(self.upper_list) != 0 \
            and len(self.lower_list) != 0:

            # head, upper, lower, masked_raw, label
            return torch.stack(self.head_list, dim=1), \
                torch.stack(self.upper_list, dim=1), \
                torch.stack(self.lower_list, dim=1), \
                torch.stack(self.masked_list, dim=0).permute(3, 0, 1, 2), \
        
        else:

            return None, None, None, None
