from __future__ import annotations

import time
from functools import cached_property

import numpy as np
from common.bases import BaseModel
from common.bases import BaseService
from common.logs import get_logger
from common.settings import Settings
from infrastructure.card_detector import CardDetector
from infrastructure.card_detector import CardDetectorInput
from infrastructure.text_detector import TextDetector
from infrastructure.text_detector import TextDetectorInput
from infrastructure.text_ocr import TextOCR
from infrastructure.text_ocr import TextOCRInput
from service.card_align import CardAlignModel

logger = get_logger(__name__)


class OCRInput(BaseModel):
    image: np.ndarray


class OCROutput(BaseModel):
    # status: bool
    results: list[list[dict]]


class OCRService(BaseService):
    settings: Settings

    @property
    def _get_card_detector(self) -> CardDetector:
        return CardDetector(settings=self.settings)

    @property
    def _get_text_detector(self) -> TextDetector:
        return TextDetector(settings=self.settings)

    @property
    def _get_text_ocr(self) -> TextOCR:
        return TextOCR(settings=self.settings)

    @cached_property
    def _card_align(self) -> CardAlignModel:
        return CardAlignModel(settings=self.settings)

    def process(self, inputs: OCRInput) -> OCROutput:
        # detect card
        try:
            start = time.perf_counter()
            card_det_output = self._get_card_detector.process(
                inputs=CardDetectorInput(
                    image=inputs.image,
                ),
            )
            logger.info(
                f'Card detection completed in {round((time.perf_counter() - start) * 1000, 2)} ms',
            )
        except Exception as e:
            logger.error(f'Failed to process card detection: {e}')
            raise e  # stop and display full error message

        results_all = []

        for bbox in card_det_output.bboxes:
            try:
                # align card
                start = time.perf_counter()
                img_processed = self._card_align.align_img(
                    img_origin=inputs.image,
                    bbox=bbox,
                )
                logger.info(
                    f'Card alignment completed for bbox {bbox} in {round((time.perf_counter() - start) * 1000, 2)} ms',
                )
            except Exception as e:
                logger.error(f'Failed to align card with bbox {bbox}: {e}')
                continue  # hoặc raise nếu muốn dừng luôn

            # detect text
            try:
                start = time.perf_counter()
                text_det_out = self._get_text_detector.process(
                    inputs=TextDetectorInput(
                        img_origin=img_processed,
                    ),
                )
                logger.info(
                    f'Text detection completed for bbox {bbox} in {round((time.perf_counter() - start) * 1000, 2)} ms',
                )
            except Exception as e:
                logger.error(
                    f'Failed to process text detection with bbox {bbox}: {e}',
                )
                continue

            # OCR
            try:
                start = time.perf_counter()
                text_ocr_out = self._get_text_ocr.process(
                    inputs=TextOCRInput(
                        img=img_processed,
                        bboxes_list=text_det_out.bboxes_list,
                        class_list=text_det_out.class_list,
                    ),
                )
                logger.info(
                    f'OCR processing completed for bbox {bbox} in {round((time.perf_counter() - start) * 1000, 2)} ms',
                )
                results_all.append(text_ocr_out.results)
            except Exception as e:
                logger.error(f'Failed to text ocr with bbox {bbox}: {e}')
                continue

        return OCROutput(results=results_all)
