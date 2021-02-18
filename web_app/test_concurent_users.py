import sys
import subprocess
import time

import angelina_reader_core

processes = 50
iters = 100

core = angelina_reader_core.AngelinaSolver()

param = sys.argv[1]

if param == "root":
    core.find_users_by_email("")
    for r in range(processes):
        subprocess.Popen(['python', 'test_concurent_users.py', str(r+1)])
    while True:
        try:
            f = core.find_users_by_email(email="b@ovdv.ru")
            print(len(f))
            if len(f) >= processes*iters:
                break
            time.sleep(1)
        except Exception as e:
            with open('static/data/a_err_.txt', 'w') as f:
                f.write(str(e) + "\n")
            raise


else:
    for i in range(iters):
        try:
            r = core.register_user(name="керогаз Керогазов", email="b@ovdv.ru", password_hash="qqwe", network_name=param, network_id=i+1)
        except Exception as e:
            with open('static/data/a_err_{}.txt'.format(param), 'w') as f:
                f.write("{} {} {}\n".format(param, i+1, str(e)))
            raise
        print("registers", param, i)



