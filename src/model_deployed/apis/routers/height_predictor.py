from __future__ import annotations

from apis.helper.exception_handler import ExceptionHandler
from apis.helper.exception_handler import ResponseMessage
from apis.models.height_predictor import APIInput
from apis.models.height_predictor import APIOutput
from common.logs import get_logger
from common.utils import get_settings
from fastapi import APIRouter
from fastapi import Body
from fastapi import status
from fastapi.encoders import jsonable_encoder
from infrastructure.height_predictor import HeightPredictorModel
from infrastructure.height_predictor import HeightPredictorModelInput

height_predictor = APIRouter(prefix='/v1')
logger = get_logger(__name__)
settings = get_settings()

# --- Load Model ---
try:
    logger.info('Load model Height Predictor!!!')
    height_pre_model = HeightPredictorModel.get_service(settings=settings)
except Exception as e:
    logger.error(f'Failed to initialize height predictor model: {e}')
    raise e


@height_predictor.post(
    '/height_pred',
    response_model=APIOutput,
    responses={
        status.HTTP_200_OK: {
            'content': {
                'application/json': {
                    'example': {
                        'message': ResponseMessage.SUCCESS,
                        'info': {'pred': [167.2, 170.5, 172.3]},
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
async def height_pre(inputs: APIInput = Body(...)):
    """
    Predict height from vector input.

    Args:
        inputs (APIInput): Contains vector x (batch of feature vectors).

    Returns:
        dict: Prediction result with 'pred' values.
    """
    exception_handler = ExceptionHandler(
        logger=logger.bind(), service_name=__name__,
    )
    logger.info('Starting height prediction...')

    # --- Validate Input ---
    try:
        if not inputs.x or not all(isinstance(row, list) for row in inputs.x):
            raise ValueError('Input vector x must be a list of lists.')
    except Exception as e:
        logger.error(f'Validation error: {e}')
        return exception_handler.handle_bad_request(str(e), jsonable_encoder(inputs))

    # --- Predict ---
    try:
        response = await height_pre_model.process(
            inputs=HeightPredictorModelInput(x=inputs.x),
        )
        api_output = APIOutput(pred=response.pred)
        logger.info('Height prediction completed.')
        return exception_handler.handle_success(jsonable_encoder(api_output))

    except Exception as e:
        logger.exception(f'Prediction error: {e}')
        return exception_handler.handle_exception(
            'Failed to perform height prediction',
            jsonable_encoder({}),
        )
