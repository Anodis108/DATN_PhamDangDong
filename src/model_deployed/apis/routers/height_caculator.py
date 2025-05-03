from __future__ import annotations

from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi.encoders import jsonable_encoder
from infrastructure.calculate import CalHeight
from infrastructure.calculate import CalHeightInput
from src.model_deployed.apis.models.height_calculator import APIInput
from src.model_deployed.apis.models.height_calculator import APIOutput

# import cv2

height_cal = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()


try:
    logger.info('Load mode Cal Height !!!')
    height_cal_model = CalHeight.get_service(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize embedding model: {e}')
    raise e  # stop and display full error message


@height_cal.post(
    '/height_cal',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {
                            'heights': [160.2, 158.6],
                            'distances': [[5.0, 10.2], [4.8, 9.9]],
                            'cm_direct': [12.4, 11.7],
                            'cm_sum': [13.5, 12.3],
                            'diffs': [1.1, 0.6],
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
async def cal_height(inputs: APIInput = Body(...)):
    """
    Performs Cal Height on the provided landmarks and image parameters.

    Args:
        inputs (APIInput): Contains landmarks, image width/height, and px_per_cm.

    Returns:
        dict: Cal Height results with height and related measurements.

    Raises:
        HTTPException: If an error occurs during Cal Height processing.
    """
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )

    logger.info('Starting Cal Height processing...')

    # --- Combined Input Validation ---
    try:
        if not inputs.landmarks or not isinstance(inputs.landmarks, list):
            raise ValueError('Invalid or missing landmarks.')
        if inputs.img_width <= 0 or inputs.img_height <= 0:
            raise ValueError('Image dimensions must be greater than zero.')
        if inputs.px_per_cm <= 0:
            raise ValueError(
                'Pixels per centimeter must be greater than zero.',
            )
    except Exception as e:
        logger.error(f'Input validation error: {e}')
        return exception_handler.handle_bad_request(str(e), jsonable_encoder(inputs))

    # --- Main Processing ---
    try:
        response = await height_cal_model.process(
            inputs=CalHeightInput(
                landmarks=inputs.landmarks,
                img_width=inputs.img_width,
                img_height=inputs.img_height,
                px_per_cm=inputs.px_per_cm,
            ),
        )

        if not response or not response.heights:
            raise ValueError('Height calculation returned no results.')

        api_output = APIOutput(
            heights=response.heights,
            distances=response.distances,
            cm_direct=response.cm_direct,
            cm_sum=response.cm_sum,
            diffs=response.diffs,
        )

        logger.info('Cal Height processing completed successfully.')
        return exception_handler.handle_success(jsonable_encoder(api_output))

    except Exception as e:
        logger.exception(f'Error during Cal Height processing: {e}')
        return exception_handler.handle_exception(
            'Failed to process Cal Height',
            jsonable_encoder({}),
        )
