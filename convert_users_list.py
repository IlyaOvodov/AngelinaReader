#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Одноразовый скрипт для конвертации старого формата all_users.json в новый
"""
import json
import uuid

fn = r"web_app\static\data\all_users.json"

with open(fn, encoding='utf-8') as f:
    all_users = json.load(f)

new_all_users = dict()

for user_key, user_dict in all_users.items():
    if "email" in user_dict.keys():
        new_key, new_dict = user_key, user_dict
    else:
        assert len(user_dict.keys()) == 1, user_dict
        assert list(user_dict.keys())[0] == "name", user_dict
        new_key = uuid.uuid4().hex
        new_dict = {
            "name": user_dict["name"],
            "email": user_key
        }
    assert new_key not in new_all_users.keys(), (new_key, new_all_users)
    new_all_users[new_key] = new_dict

with open(fn, 'w', encoding='utf8') as f:
    json.dump(new_all_users, f, sort_keys=True, indent=4, ensure_ascii=False)
