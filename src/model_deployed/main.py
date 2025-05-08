from __future__ import annotations

from apis.helper import LoggingMiddleware
from apis.routers.box_detector import box_detector
from apis.routers.height_caculator import height_cal
from apis.routers.height_predictor import height_predictor
from apis.routers.pose_detector import pose_detector
from asgi_correlation_id import CorrelationIdMiddleware
from common.logs import get_logger
from common.logs import setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

setup_logging(json_logs=False)
logger = get_logger('api')

app = FastAPI(title='Model Deployed API - AI cal height', version='1.0.0')


# add middleware to generate correlation id
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(CorrelationIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(
    box_detector,
)

app.include_router(
    pose_detector,
)

app.include_router(
    height_cal,
)

app.include_router(
    height_predictor,
)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=5000, reload=True)
