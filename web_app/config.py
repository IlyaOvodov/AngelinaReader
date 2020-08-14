import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'angilina'
    DATA_ROOT = os.environ.get('DATA_ROOT') or 'static/data'
    # e-mail parameters:
    SMTP_SERVER = os.environ.get('SMTP_SERVER') or "adamant.host27.info"
    SMTP_PORT = os.environ.get('SMTP_PORT') or "587"
    SMTP_PWD = os.environ.get('SMTP_PWD') or "w!i[CRuUG4,)"
    SMTP_FROM = os.environ.get('SMTP_FROM') or "results@angelina-reader.ru"
