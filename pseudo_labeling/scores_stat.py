from pathlib import Path
import local_config
import data_utils.data as data

original_path = Path(local_config.data_path) / 'AngelinaDataset'

original_list = 'handwritten/train.txt'
pseudolab_root = 'pseudo1'

pseudolab_path = original_path / pseudolab_root / Path(original_list).parent
with open(original_path / original_list) as f:
    files = f.readlines()
mean_scores = []
for fn in files:
    full_fn = pseudolab_path / fn.strip().replace('\\', '/')

    assert full_fn.is_file(), full_fn
    rects = None
    lbl_fn = full_fn.with_suffix('.json')
    assert lbl_fn.is_file(), lbl_fn
    rects = data.read_LabelMe_annotation(label_filename=lbl_fn, get_points=False)
    if rects is not None:
        boxes = [r[:4] for r in rects]
        labels = [r[4] for r in rects]
        scores = [r[5] for r in rects]
        mean_score = sum(scores)/len(scores)
        s = [mean_score]
        scores = sorted(scores)   # incompatible with boxes, labels
        s += [scores[ (k * len(scores))//100] for k in (1, 5,25,50)]
        mean_scores.append(s)
        #print(mean_score)
for i, v in enumerate(sorted(mean_scores)):
    print(i, [int(vi*1000) for vi in v])

