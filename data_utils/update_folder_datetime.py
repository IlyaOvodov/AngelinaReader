import os
from pathlib import Path

root = r'E:\_ELVEES\Braille'

def update(root):
    maxtime = 0
    for sub in os.listdir(root):
        sub = os.path.join(root, sub)
        if os.path.isdir(sub):
            update(sub)
        maxtime = max(maxtime, os.path.getmtime(sub))
    os.utime(root, (maxtime, maxtime))

if __name__=='__main__':
    update(root)