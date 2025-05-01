from __future__ import annotations

import cv2
import numpy as np
from api.helper.exception_handler import ExceptionHandler
from api.helper.exception_handler import ResponseMessage
from api.models.ocr_card import APIOutput
from app.ocr import OCRInput
from app.ocr import OCRService
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import File
from fastapi import status
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder


ocr = APIRouter(prefix='/v1')
logger = get_logger(__name__)

settings = get_settings()


# Define API input


@ocr.post(
    '/ocr',
    # response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'info_text': [
                                {
                                    'class_name': 'name',
                                    'bounding_box': [100.0, 200.0, 300.0, 400.0],
                                    'text': 'Nguyen Van A',
                                },
                            ],
                        },
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            'description': 'Bad Request',
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.BAD_REQUEST,
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
async def ocr_card(file: UploadFile = File(...)):
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )

    try:
        logger.info('Received OCR request', extra={'file_name': file.filename})
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_array is None:
            raise ValueError('Failed to decode image - result is None')
    except Exception as e:
        return exception_handler.handle_exception(
            err_msg=f'Error while reading and decoding file: {e}',
            details={'file_name': file.filename},
        )
    # Define application
    try:
        logger.info(
            'Initializing OCR model...',
            extra={'file_name': file.filename},
        )
        ocr_model = OCRService(settings=settings)
        logger.info(
            'OCR model initialized successfully',
            extra={'file_name': file.filename},
        )
    except Exception as e:
        return exception_handler.handle_exception(
            f'Failed to initialize OCR model: {e}',
            details={'file_name': file.filename},
        )
    # infer
    try:
        logger.info(
            'Running OCR inference...', extra={
                'file_name': file.filename,
            },
        )
        text_ocr_result = ocr_model.process(
            inputs=OCRInput(image=img_array),
        )

        api_output = APIOutput(info_text=text_ocr_result.results)
        return exception_handler.handle_success(jsonable_encoder(api_output))

    except Exception as e:
        return exception_handler.handle_exception(
            f'OCR processing failed: {e}',
            details={'file_name': file.filename},
        )
