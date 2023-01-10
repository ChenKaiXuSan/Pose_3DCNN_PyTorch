# %%
import math
import warnings

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from torchvision.io import read_image, read_video, write_video
from torchvision.utils import save_image
from torchvision.transforms.functional import crop, resize

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
        annotated_image = image.permute(1, 2, 0).cpu().numpy().copy()

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
                        keypoint
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
                        ear_center_point[0] * width - 20),
                        height=int(shoulder_center_point[1] * height), width=int(shoulder_center_point[1] * height))
        head_img = resize(head_img, [self.img_size, self.img_size])

        return head_img

    def split_upper_part(
        self,
        image: torch.Tensor,
        keypoint
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

        body_img = crop(image.permute(2, 0, 1),
                        top=int(shoulder_center_point[1] * height),
                        left=int(shoulder_center_point[0] * width - 20),
                        height=int(body_height), width=int(body_height))
        body_img = resize(body_img, [self.img_size, self.img_size])


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

        lower_img = crop(image.permute(2, 0, 1),
                        top=int(hip_center_point[1] * height),
                        left=int(hip_center_point[0] * width - 60),
                        height=int(lower_height), width=int(lower_height))
        lower_img = resize(lower_img, [self.img_size, self.img_size])

        save_image(lower_img/255, fp='head.png')

        return lower_img

    def handle_batch_video(self, batch):
        
        video = batch['video']
        video_name = batch['video_name']

        B, c, T, h, w = video.size()

        # store the batch infor, to return.
        batch_head_list = []
        batch_upper_list = []
        batch_lower_list = []
        batch_masked_raw_list = []

        for b in range(B):
            batch_video = video[b]

            keypoints_list = []
            masked_list = []
            head_list = []
            upper_list = []
            lower_list = []

            for i in range(T):

                masked_image, keypoints = self.mask_pose(batch_video[:, i,:])

                if masked_image is not None and keypoints is not None:
                    keypoints_list.append(keypoints)  # 1, 17, 3
                    masked_list.append(masked_image)

            if masked_list is None:
                # if not pred, drop all batch.
                continue
            else:

                for m, k in zip(masked_list, keypoints_list):
                    head_list.append(self.split_head_part(m, k))
                    upper_list.append(self.split_upper_part(m, k))
                    lower_list.append(self.split_lower_part(m, k))

                # when not kpt pred, not drop frame but copy the last frame.
                while True:
                    if len(masked_list) != self.unifrom_temporal_subsample_num:
                        masked_list.append(masked_list[-1])
                        head_list.append(head_list[-1])
                        upper_list.append(upper_list[-1])
                        lower_list.append(lower_list[-1])
                    else: 
                        break;

                batch_head_list.append(torch.stack(head_list, dim=1)) # c, t, h, w
                batch_upper_list.append(torch.stack(upper_list, dim=1)) 
                batch_lower_list.append(torch.stack(lower_list, dim=1))
                batch_masked_raw_list.append(torch.stack(masked_list, dim=0).permute(3, 0, 1, 2)) # t, h, w, c to c, t, h, w

        return torch.stack(batch_head_list, dim=0).cuda(), \
            torch.stack(batch_upper_list, dim=0).cuda(), \
            torch.stack(batch_lower_list, dim=0).cuda(), \
            torch.stack(batch_masked_raw_list, dim=0).cuda()


