from pathlib import Path
import json

root_dir = r'D:\Programming.Data\Braille\web_uploaded\re-processed200823'
count = 0
for fn in Path(root_dir).glob('**/*.json'):
    lbl = json.load(open(fn))
    lbl['imageData'] = None
    is_changed = False

    for s in lbl['shapes']:
        if s["label"] == "'":
            print(str(fn), s)
            s["label"] = "~4"
            is_changed = True
    if is_changed:
        count += 1
    #     json.dump(lbl, open(fn, mode='w'), indent=2)
print("corrected: ", count)