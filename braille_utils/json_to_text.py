import argparse
import os.path
import glob
import sys
sys.path.insert(1, '..')
import braille_utils.postprocess as postprocess
import train.data as data


def annonation_to_text(json_filename, lang):
    rects = data.read_LabelMe_annotation(label_filename = json_filename, get_points = False)
    boxes = [r[:4] for r in rects]
    labels = [r[4] for r in rects]
    lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
    return postprocess.lines_to_text(lines)


def process_json_annotation(json_filename, lang='RU'):
    print('processing ' + json_filename)
    txt = annonation_to_text(json_filename, lang)
    if json_filename.endswith('.labeled.json'):
        out_fn = json_filename[:-len('.labeled.json')]+'.marked.txt'
    else:
        raise Exception('incorrect filename')
    with open(out_fn, 'wt') as f:
        f.write(txt)

if __name__ == "__main__":
    parcer = argparse.ArgumentParser()
    parcer.add_argument('file')
    args = parcer.parse_args()
    print(args.file)
    if os.path.isfile(args.file) and args.file.lower().endswith('.json'):
        process_json_annotation(args.file)
    elif os.path.isdir(args.file):
        files = glob.glob(os.path.join(args.file, '*.json'))
        for fn in files:
            process_json_annotation(fn)
    else:
        raise Exception('incorrect argument: ' + args.file)
