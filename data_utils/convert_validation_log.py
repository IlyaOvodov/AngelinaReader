"""
convert log produiced by validate_retinanet
"""
input_files = [
    r"D:\Programming.Data\Braille\validate_retinanet_normal.log",
    r"D:\Programming.Data\Braille\validate_retinanet_flipped.log",
    ]

re_template = r"^NN_results\\210811_flip_test\\(?:\d{6})_flip_test(?P<track1>(?:_\d+)+)\\(?P<train_set>angelina|dsbi)(?:_21)?_fpn1_lay4(?P<track2>(?:_\d+)+)_V(?P<aug_flip>False|True)_(?<id>[\da-f]+)\smodels\\clr\.(?P<clr_no>\d{3})\.t7\s+(?P<test_set>dsbi|Angelina)\s+[\d+-\.e]+" \
              "\s+(?P<v1>[\d\.e+-]+)\s+(?P<v2>[\d\.e+-]+)\s+(?P<v3>[\d\.e+-]+)\s+(?P<v4>[\d\.e+-]+)\s+(?P<v5>[\d\.e+-]+)\s+(?P<v6>[\d\.e+-]+)\s+(?P<v7>[\d\.e+-]+)\s+(?P<rest>.*)$"
write_original = False  # for debugging

from pathlib import Path
import regex as re

def convert(ln, test_on_flipped):
    res = re.match(re_template, ln)
    # track1 base track2
    assert res is not None, (ln,"\n", res)
    track1 = res.group("track1")
    assert track1
    train_set = res.group("train_set")
    assert train_set
    track2 = res.group("track2")
    assert track2
    aug_flip = res.group("aug_flip")
    assert aug_flip
    id = res.group("id")
    assert id
    clr_no = res.group("clr_no")
    assert clr_no
    test_set = res.group("test_set")
    assert test_set
    v1 = res.group("v1")
    v2 = res.group("v2")
    v3 = res.group("v3")
    v4 = res.group("v4")
    v5 = res.group("v5")
    v6 = res.group("v6")
    v7 = res.group("v7")

    rest = res.group("rest")
    assert rest == ""

    if track1 == "_0":
        assert track2 == "_1"
    else:
        assert track1 == track2
    assert track1[0] == "_"
    del track2

    prev_depth = len(track1.split("_")) - 2  #_N -> ["", "N"] -> 0
    prev_clr = prev_depth*10

    clr_no = int(clr_no)
    assert clr_no > 0 and clr_no < 100
    full_clr_no = prev_clr + clr_no

    res_ln = f"{track1}\t{train_set}\t{aug_flip}\t{id}\t{clr_no}\t{prev_depth}\t{full_clr_no}\t{test_set}\t{test_on_flipped}" \
        f"\t{v1}\t{v2}\t{v3}\t{v4}\t{v5}\t{v6}\t{v7}\n"

    return res_ln

def main(input_file):
    test_on_flipped = "flipped" in Path(input_file).name
    out_file = input_file + ".txt"
    with open(input_file) as f:
        lines = f.readlines()
    assert lines[0].startswith("model	weights	key	time	precision	recall	f1	precision_c	recall_C	f1_c	cer(%)"), lines[0]
    lines = lines[1:]
    with open(out_file, "w") as f:
        f.write("track\ttrain_set\taug_flip\tid\tclr_no\tprev_depth\tfull_clr_no\ttest_set\ttest_on_flipped" \
        f"\tprecision\trecall\tf1\tprecision_c\trecall_c\tf1_c\tcer(%)\n")
        for ln in lines:
            if write_original:
                f.write(ln)
            res_ln = convert(ln, test_on_flipped)
            f.write(res_ln)


if __name__ == "__main__":
    for fn in input_files:
        main(fn)
