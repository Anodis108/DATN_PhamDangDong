from __future__ import annotations

import cv2
import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.box_detector import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import File
from fastapi import status
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from infrastructure.box_detector import BoxDetectorModel
from infrastructure.box_detector import BoxDetectorModelInput

box_detector = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Box detector !!!')
    box_detector_model = BoxDetectorModel(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize Box embedding model: {e}')
    raise e  # stop and display full error message


@box_detector.post(
    '/box_detector',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'bboxes': [
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                            ],
                            'scores': [1, 0.5],
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
async def box_detect(file: UploadFile = File(...)):
    """
    Detects Boxs in the provided input data.

    Args:
        inputs (BoxDetectorInput): The input data for Box detection, which includes image information.
    Returns:
        BoxDetectorOutput: The output data containing detected Boxs and related extra.
    Raises:
        HTTPException: If an error occurs during Box detection processing.
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
    except Exception as e:
        return exception_handler.handle_exception(
            err_msg=f'Error while reading file: {e}',
            extra={'file_name': file.filename},
        )

    try:
        # Process image
        response = await box_detector_model.process(
            inputs=BoxDetectorModelInput(
                img=img_array,
            ),
        )
        # handle response
        api_output = APIOutput(
            bboxes=response.bboxes.tolist(),  # đảm bảo trả về dạng list[list]
            scores=response.scores.tolist(),  # nếu có scores
            pixel_per_cm=response.pixel_per_cm,
        )
        return exception_handler.handle_success(jsonable_encoder(api_output))
    except Exception as e:
        return exception_handler.handle_exception(
            err_msg=f'Error during Box detection: {e}',
            extra={'input': file.filename},
        )
