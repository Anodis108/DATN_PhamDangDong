from __future__ import annotations

import cv2
import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.pose_detector import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import File
from fastapi import status
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from infrastructure.pose_detector import PoseDetectorModel
from infrastructure.pose_detector import PoseDetectorModelInput

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
    '/pose_detector',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'pose_landmarks': [
                            [
                                {'x': 0.5, 'y': 0.3, 'z': 0.1, 'visibility': 0.9},
                                {'x': 0.6, 'y': 0.4, 'z': 0.2, 'visibility': 0.8},
                            ],
                        ],
                        'img_width': 640,
                        'img_height': 480,
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': 'Bad Request - Invalid image data',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
                        'error': 'Invalid image format',
                    },
                },
            },
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            'description': 'Internal Server Error',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.INTERNAL_SERVER_ERROR,
                        'error': 'Failed to process Pose detection',
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
                        'error': 'Unsupported image format',
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
                        'error': 'Resource not found',
                    },
                },
            },
        },
    },
)
async def pose_detect(file: UploadFile = File(...)):
    """
    Detects Poses in the provided image file.

    Args:
        file (UploadFile): The input image file for pose detection (e.g., JPEG, PNG).
    Returns:
        APIOutput: The output data containing detected pose landmarks and image dimensions.
    Raises:
        HTTPException: If an error occurs during pose detection processing.
    """
    # Validate input parameters
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )
    try:
        contents = await file.read()

        # Chuyển dữ liệu ảnh thành mảng numpy
        nparr = np.frombuffer(contents, np.uint8)

        # Giải mã ảnh thành định dạng OpenCV (BGR)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            return exception_handler.handle_bad_request(
                err_msg='Invalid image format',
                extra={'file_name': file.filename},
            )
    except Exception as e:
        return exception_handler.handle_exception(
            err_msg=f'Error while reading file: {e}',
            extra={'file_name': file.filename},
        )

    try:
        # Process image
        response = await pose_detector_model.process(
            inputs=PoseDetectorModelInput(
                img=img_array,
            ),
        )
        serialized_landmarks = [
            [
                {'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks
            ] for landmarks in response.pose_landmarks
        ]

        # Handle response
        api_output = APIOutput(
            pose_landmarks=serialized_landmarks,
            img_width=response.img_width,
            img_height=response.img_height,
        )
        return exception_handler.handle_success(jsonable_encoder(api_output))
    except Exception as e:
        return exception_handler.handle_exception(
            err_msg=f'Error during Pose detection: {e}',
            extra={'input': file.filename},
        )
