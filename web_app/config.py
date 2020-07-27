import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'angilina'
    DATA_ROOT = os.environ.get('DATA_ROOT') or 'static/data'