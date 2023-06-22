import json
from pathlib import Path

def process_file(input_file, output_file):

    with open(input_file) as f:
        templates = json.load(f)

    for i in range(len(templates["shapes"])):
        templates["shapes"][i].pop("line_color")
        templates["shapes"][i].pop("fill_color")
        if "flags" in templates["shapes"][i]:
            templates["shapes"][i].pop("flags")
    if "imageData" in templates:
        templates.pop("imageData")
    if "lineColor" in templates:
        templates.pop("lineColor")
    if "flags" in templates:
        templates.pop("flags")
    if "fillColor" in templates:
        templates.pop("fillColor")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(templates, f, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
    root_dir = "/home/orwell/brail/rescontrast/neural/"
    dest_dir = "/home/orwell/brail/rescontrast/neuralout"
    files = Path(root_dir).glob("**/*.json")
    for fn in files:
        rel_path = fn.relative_to(root_dir)
        print(rel_path)
        process_file(fn, dest_dir/rel_path)
