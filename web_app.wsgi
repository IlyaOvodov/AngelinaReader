#!/usr/bin/python
import sys
import logging
from pathlib import Path
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,str(Path(__file__).parent))
from web_app.angelina_reader_app import app as application
