import os
import datetime

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'angilina'
    MODEL_PATH = "."
    DATA_ROOT = os.environ.get('DATA_ROOT') or 'static/data'
    # e-mail parameters:
    SMTP_SERVER = os.environ.get('SMTP_SERVER') or "ovdv.ru"
    SMTP_PORT = os.environ.get('SMTP_PORT') or "587"
    SMTP_PWD = os.environ.get('SMTP_PWD') or ""
    SMTP_FROM = os.environ.get('SMTP_FROM') or "results@angelina-reader.ru"
    PERMANENT_SESSION_LIFETIME = datetime.timedelta(minutes=60*24*365*2)