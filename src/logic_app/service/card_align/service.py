from __future__ import annotations

from typing import List

import cv2
import numpy as np
from common.bases import BaseModel
from common.settings import Settings

# import json
# from functools import cached_property


class CardAlignModel(BaseModel):
    settings: Settings

    def align_img(self, img_origin: np.ndarray, bbox: List[int]) -> np.ndarray:
        # croping img
        x1, y1, x2, y2 = map(int, bbox)  # Lấy tọa độ bounding box
        img2 = img_origin[y1:y2, x1:x2]

        # Loading image using cv2
        baseImg = cv2.imread(self.settings.card_align.baseimg_path)
        # Declare image size, width height and chanel
        baseH, baseW, baseC = baseImg.shape

        # Init orb, keypoints detection on base Image
        orb = cv2.ORB_create(1000)

        kp, des = orb.detectAndCompute(baseImg, None)
        # imgKp = cv2.drawKeypoints(baseImg,kp, None)

        # Detect keypoint on img2
        kp1, des1 = orb.detectAndCompute(img2, None)

        # Init BF Matcher, find the matches points of two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = list(bf.match(des1, des))

        # Select top 30% best matcher
        matches.sort(key=lambda x: x.distance)
        best_matches = matches[
            :int(
                len(matches)*self.settings.card_align.per_match,
            )
        ]

        # Init source points and destination points for findHomography function.
        srcPoints = np.float32(
            [kp1[m.queryIdx].pt for m in best_matches],
        ).reshape(-1, 1, 2)
        dstPoints = np.float32(
            [kp[m.trainIdx].pt for m in best_matches],
        ).reshape(-1, 1, 2)

        # Find Homography of two images
        matrix_relationship, _ = cv2.findHomography(
            srcPoints, dstPoints, cv2.RANSAC, 5.0,
        )

        # Transform the image to have the same structure as the base image
        img_final = cv2.warpPerspective(
            img2, matrix_relationship, (baseW, baseH),
        )

        return img_final

    def process2white_black(self, img: np.ndarray) -> np.ndarray:
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.multiply(gray, 1.5)

        # blur remove noise
        blured1 = cv2.medianBlur(gray, 3)
        blured2 = cv2.medianBlur(gray, 51)
        divided = np.ma.divide(blured1, blured2).data
        normed = np.uint8(255*divided/divided.max())

        # Threshold image
        th, threshed = cv2.threshold(
            normed, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY,
        )

        return threshed
