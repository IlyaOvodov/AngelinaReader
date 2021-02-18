from collections import defaultdict
import json
from pathlib import Path

files = Path(r"static\data\results").glob("*.protocol.txt")

files = list(files)
print(len(files), "files")

user_map = defaultdict(int)
for fn in files:
    with open(fn) as f:
        prot = json.load(f)
        user_id = prot['user']
        user_map[user_id] += 1
best_users = [(v,k) for k,v in user_map.items()]
best_users.sort(reverse=True)
print(len(best_users), 'total')
prev_n = 0
for i, x in enumerate(best_users):
    if x[0] != prev_n and i>0:
        print(i-1, best_users[i-1][0])
        prev_n = x[0]



