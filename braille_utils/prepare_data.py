import local_config
import os
import data_utils.data as data
import braille_utils.postprocess as postprocess

lang = 'RU'

datasets = {
    #'test': [r'AngelinaDataset/books/train.txt'],
    'val': [r'AngelinaDataset/handwritten/train.txt'],
}

def prepare_data(datasets=datasets):
    """
    data (datasets defined above as global) -> dict: key - list of dict (image_fn":full image filename, "gt_text": groundtruth pseudotext, "gt_rects": groundtruth rects + label 0..64)
    :return:
    """
    min_align_score = 0
    res_dict = dict()
    for key, list_file_names in datasets.items():
        data_list = list()
        res_dict[key] = data_list
        for list_file_name in list_file_names:
            list_file = os.path.join(local_config.data_path, list_file_name)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            for fn in files:
                if fn[-1] == '\n':
                    fn = fn[:-1]
                fn = fn.replace('\\', '/')
                full_fn = os.path.join(data_dir, fn)
                if os.path.isfile(full_fn):
                    rects = None
                    lbl_fn = full_fn.rsplit('.', 1)[0] + '.json'
                    if os.path.isfile(lbl_fn):
                        rects = data.read_LabelMe_annotation(label_filename=lbl_fn, get_points=False)
                        boxes = [r[:4] for r in rects]
                        labels = [r[4] for r in rects]
                        scores = [r[5] for r in rects]
                        lines = postprocess.boxes_to_lines(boxes, labels, scores=scores, lang=lang, filter_lonely=False,
                                                           min_align_score=min_align_score)
                        txt =  postprocess.lines_to_text(lines)
                        if lbl_fn.endswith('.labeled.json'):
                            out_fn = lbl_fn[:-len('.labeled.json')] + '.marked.txt'
                        else:
                            raise Exception('incorrect filename')
                        with open(out_fn, 'wt') as f:
                            f.write(txt)
                    #if rects is not None:
                        #data_list.append({"image_fn":full_fn, "gt_rects": rects})
    #return res_dict

def main():
   prepare_data()

if __name__ == '__main__':
    main()