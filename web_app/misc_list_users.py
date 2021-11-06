#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from flask import Flask

import os
import json
import sys
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

users_file = app.config['DATA_ROOT'] + '/all_users.json'
if os.path.isfile(users_file):
    with open(users_file) as f:
        all_users = json.load(f)
else:
    all_users = dict()

for e_mail, d in all_users.items():
    print(e_mail, d)