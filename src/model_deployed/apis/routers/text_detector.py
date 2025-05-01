from __future__ import annotations

import time

import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.text_detector import APIInput
from apis.models.text_detector import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi.encoders import jsonable_encoder
from infrastructure.pose_detector import TextDetectorModel
from infrastructure.pose_detector import TextDetectorModelInput
# import cv2

text_detector = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Text detector !!!')
    text_detector_model = TextDetectorModel(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize Text embedding model: {e}')
    raise e  # stop and display full error message


@text_detector.post(
    '/text_detector',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'classes': ['birth', 'name'],
                            'bboxes': [
                                [1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0, 2.0],
                            ],
                            'confs': [1.0, 0.5],
                            'processed_images': [
                                [[0, 0, 0], [255, 255, 255]],
                                [[128, 128, 128], [64, 64, 64]],
                            ],
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': 'Bad Request - message is required',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': 'Internal Server Error - Error during init conversation',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                    },
                },
            },
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            'description': 'Unprocessable Entity - Format is not supported',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.UNPROCESSABLE_ENTITY,
                    },
                },
            },
        },
        status.HTTP_404_NOT_FOUND: {
            'description': 'Destination Not Found',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.NOT_FOUND,
                    },
                },
            },
        },
    },
)
async def text_detect(inputs: APIInput = Body(...)):
    """
    Detects texts in the provided image.

    Args:
        inputs (APIInput): Input containing image data.

    Returns:
        JSON response containing detected texts and bounding boxes.
    """
    start_total = time.perf_counter()
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )

    if inputs is None or not inputs.image:
        return exception_handler.handle_bad_request(
            'Invalid image data',
            jsonable_encoder(inputs),
        )

    try:
        logger.info('Processing text detection ...')
        logger.info(
            'np.array(inputs.image)',
            shape=np.array(inputs.image).shape,
        )

        # text detector
        t6 = time.perf_counter()
        response = await text_detector_model.process(
            inputs=TextDetectorModelInput(
                img_processed=np.array(inputs.image, dtype=np.uint8),
            ),
        )
        t7 = time.perf_counter()
        logger.info(f'[Timer] Model processing: {(t7 - t6)*1000:.2f} ms')

        t8 = time.perf_counter()
        # handle response
        api_output = APIOutput(
            bboxes=response.bboxes_list,
            classes=response.class_list,
            confs=response.conf_list,
        )
        t9 = time.perf_counter()
        logger.info(f'[Timer] Build response model: {(t9 - t8)*1000:.2f} ms')

        logger.info('Text detection completed successfully.')

        total_time = (time.perf_counter() - start_total) * 1000
        logger.info(f'[Timer] Total API time: {total_time:.2f} ms')

        return exception_handler.handle_success(jsonable_encoder(api_output))

    except ValueError as ve:
        return exception_handler.handle_bad_request(str(ve), jsonable_encoder(inputs))

    except TypeError as te:
        return exception_handler.handle_bad_request(str(te), jsonable_encoder(inputs))

    except FileNotFoundError as fnf:
        return exception_handler.handle_not_found_error(str(fnf), jsonable_encoder(inputs))

    except RuntimeError as re:
        return exception_handler.handle_exception(str(re), jsonable_encoder(inputs))

    except Exception as e:
        logger.exception(
            f'Exception occurred while processing text detection: {e}',
        )
        return exception_handler.handle_exception(
            'Failed to process text detection',
            jsonable_encoder(inputs),
        )
