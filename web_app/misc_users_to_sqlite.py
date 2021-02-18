import json
from pathlib import Path
import sqlite3

import angelina_reader_core

json_file = "static/data/all_users.json"
db_file = Path(json_file).with_suffix(".db")

with open(json_file, encoding='utf-8') as f:
    all_users = json.load(f)

if db_file.is_file():
    con = sqlite3.connect(str(db_file))
else:
    con = sqlite3.connect(str(db_file))
    con.cursor().execute("CREATE TABLE users(id text PRIMARY KEY, name text, email text, network_name text, network_id text, password_hash text, reg_date text)")
    con.commit()

for id, user_dict in all_users.items():
    con.cursor().execute("INSERT INTO users(id, name, email) VALUES(?, ?, ?)", (id, user_dict["name"], user_dict["email"]))
con.commit()

