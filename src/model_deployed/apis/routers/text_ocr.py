from __future__ import annotations

import numpy as np
from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.text_ocr import APIInput
from apis.models.text_ocr import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi.encoders import jsonable_encoder
from infrastructure.text_ocr import TextOCRModel
from infrastructure.text_ocr import TextOCRModelInput

# import cv2

text_ocr = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Text OCR !!!')
    text_ocr_model = TextOCRModel(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize Text embedding model: {e}')
    raise e  # stop and display full error message


@text_ocr.post(
    '/text_ocr',
    response_model=APIOutput,
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
async def text_to_ocr(inputs: APIInput = Body(...)):
    """
    Performs OCR on the provided input image.

    Args:
        inputs (APIInput): Contains image data and optional bounding boxes.
    Returns:
        dict: OCR results containing detected text and related details.
    Raises:
        HTTPException: If an error occurs during OCR processing.
    """
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )

    # Validate input
    if inputs is None or not inputs.img:
        return exception_handler.handle_bad_request(
            'Invalid image data',
            jsonable_encoder(inputs),
        )

    try:
        # Convert input image to numpy array
        img_array = np.array(inputs.img, dtype=np.uint8)
        bbox_np = np.array(inputs.bboxes, dtype=np.int32)

        # Log processing start
        logger.info('Starting OCR processing...')

        # Process image with OCR model
        response = await text_ocr_model.process(
            inputs=TextOCRModelInput(
                img=img_array,
                class_list=inputs.classes,
                bboxes_list=bbox_np.tolist(),
            ),
        )

        # Check if OCR found any text
        if not response.results:
            return exception_handler.handle_bad_request(
                'No text detected in the image.',
                jsonable_encoder(inputs),
            )

        # Format response
        info = [
            {
                'class_name': r['class'],
                'bounding_box': r['bounding_box'],
                'text': r['text'],
            }
            for r in response.results
        ]

        api_output = APIOutput(info_text=info)

        logger.info('OCR processing completed successfully.')
        return exception_handler.handle_success(jsonable_encoder(api_output))

    except Exception as e:
        logger.exception(f'Error during OCR processing: {e}')
        return exception_handler.handle_exception(
            'Failed to process OCR',
            jsonable_encoder({}),
        )
