from __future__ import annotations

import cv2
import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.card_detector import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import File
from fastapi import status
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from infrastructure.box_detector import CardDetectorModel
from infrastructure.box_detector import CardDetectorModelInput

card_detector = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Card detector !!!')
    card_detector_model = CardDetectorModel(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize Card embedding model: {e}')
    raise e  # stop and display full error message


@card_detector.post(
    '/card_detector',
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
async def card_detect(file: UploadFile = File(...)):
    """
    Detects Cards in the provided input data.

    Args:
        inputs (CardDetectorInput): The input data for Card detection, which includes image information.
    Returns:
        CardDetectorOutput: The output data containing detected Cards and related details.
    Raises:
        HTTPException: If an error occurs during Card detection processing.
    """
    # Validate input parameters
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )
    try:
        # Đọc dữ liệu ảnh từ UploadFile
        contents = await file.read()

        # Chuyển dữ liệu ảnh thành mảng numpy
        nparr = np.frombuffer(contents, np.uint8)

        # Giải mã ảnh thành định dạng OpenCV (BGR)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error while reading file: {e}',
            extra={'file_name': file.filename},
        )

    try:
        # Process image
        response = await card_detector_model.process(
            inputs=CardDetectorModelInput(
                img=img_array,
            ),
        )
        # handle response
        api_output = APIOutput(
            bboxes=response.bboxes.tolist(),  # đảm bảo trả về dạng list[list]
            scores=response.scores.tolist(),  # nếu có scores
        )
        return exception_handler.handle_success(jsonable_encoder(api_output))
    except Exception as e:
        return exception_handler.handle_exception(
            e=f'Error during Card detection: {e}',
            extra={'input': file.filename},
        )
