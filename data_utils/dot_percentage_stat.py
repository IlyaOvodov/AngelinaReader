from pathlib import Path
import dsbi
import json
import braille_utils.label_tools as lt

#DSBI
def eval_dsbi():
    root_dir = Path(r"D:\Programming.Data\Braille\DSBI\data")

    txt_files = root_dir.glob('*/*+*+*.txt')

    sum_ones = 0
    sum_total = 0
    docs = 0
    for fn in txt_files:
        _, _, _, cells = dsbi.read_txt(fn)
        if cells is not None:
            docs += 1
            for cell in cells:
                if cell.label != '000000':
                    n_ones = len(cell.label.replace('0', ''))
                    sum_ones += n_ones
                    sum_total += 6
    print('dsbi', docs, sum_ones, sum_total, sum_ones/sum_total)

def eval_angelina():
    root_dir = Path(r"D:\Programming.Data\Braille\AngelinaDataset\books")

    txt_files = root_dir.glob('*/*.json')

    sum_ones = 0
    sum_total = 0
    docs = 0
    for fn in txt_files:
        docs += 1
        with open(fn, 'r', encoding='cp1251') as opened_json:
            loaded = json.load(opened_json)
        for shape in loaded["shapes"]:
            try:
                label = lt.int_to_label010(lt.human_label_to_int(shape["label"]))
                n_ones = len(label.replace('0', ''))
                sum_ones += n_ones
                sum_total += 6
            except:
                pass
    print('angelina', docs, sum_ones, sum_total, sum_ones / sum_total)

eval_dsbi()
eval_angelina()


