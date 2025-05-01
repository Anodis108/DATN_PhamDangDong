from __future__ import annotations

from apis.helper import LoggingMiddleware
from apis.routers.card_detector import card_detector
from apis.routers.text_detector import text_detector
from apis.routers.text_ocr import text_ocr
from asgi_correlation_id import CorrelationIdMiddleware
from common.logs import get_logger
from common.logs import setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

setup_logging(json_logs=False)
logger = get_logger('api')

app = FastAPI(title='Model Deployed API - AI Card Checkin', version='1.0.0')


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
    card_detector,
)

app.include_router(
    text_detector,
)

app.include_router(
    text_ocr,
)
