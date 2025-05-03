from __future__ import annotations

import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi.encoders import jsonable_encoder
from infrastructure.pose_detector import PoseDetectorModel
from infrastructure.pose_detector import PoseDetectorModelInput
from src.model_deployed.apis.models.pose_detector import APIInput
from src.model_deployed.apis.models.pose_detector import APIOutput
# import cv2

pose_detector = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Pose detector !!!')
    pose_detector_model = PoseDetectorModel(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize Pose embedding model: {e}')
    raise e  # stop and display full error message


@pose_detector.post(
    '/Pose_detector',
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
async def pose_detect(inputs: APIInput = Body(...)):
    """
    Detects Poses in the provided image.

    Args:
        inputs (APIInput): Input containing image data.

    Returns:
        JSON response containing detected Poses and bounding boxes.
    """
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )

    if inputs is None or not inputs.image:
        return exception_handler.handle_bad_request(
            'Invalid image data',
            jsonable_encoder(inputs),
        )

    try:
        logger.info('Processing Pose detection ...')

        # Pose detector
        response = await pose_detector_model.process(
            inputs=PoseDetectorModelInput(
                img_processed=np.array(inputs.image, dtype=np.uint8),
            ),
        )
        # handle response
        api_output = APIOutput(
            pose_landmarks=response.pose_landmarks,
            img_width=response.img_width,
            img_height=response.img_height,
        )

        logger.info('Pose detection completed successfully.')

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
            f'Exception occurred while processing Pose detection: {e}',
        )
        return exception_handler.handle_exception(
            'Failed to process Pose detection',
            jsonable_encoder(inputs),
        )
