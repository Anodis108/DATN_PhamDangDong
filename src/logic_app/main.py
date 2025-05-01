from __future__ import annotations

from api.helper import LoggingMiddleware
from api.routers.ocr_card import ocr
from asgi_correlation_id import CorrelationIdMiddleware
from common.logs import get_logger
from common.logs import setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from api.routers.sign_up import sign_up_endpoint

setup_logging(json_logs=False)
logger = get_logger('api')

app = FastAPI(title='OCR API - AI OCR', version='2.0.0')


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
    ocr,
)
