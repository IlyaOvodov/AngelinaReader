import collections

CellInfo = collections.namedtuple('CellInfo', 
                                  ['row', 'col',
                                   'left', 'top', 'right', 'bottom',
                                   'label'])

def read_txt(file_txt, binary_label = True):
    with open(file_txt, 'r') as f:
        l = f.readlines()
        if len(l) < 3:
            return None, None, None, None
        angle = eval(l[0])
        v_lines = list(map(eval, l[1].split(' ')))
        assert len(v_lines)%2 == 0, (file_txt, len(v_lines))
        h_lines = list(map(eval, l[2].split(' ')))
        assert len(h_lines)%3 == 0, (file_txt, len(h_lines))
        cells = []
        for cell_ln in l[3:]:
            cell_nums = list(cell_ln[:-1].split(' ')) # exclude last '\n'
            assert len(cell_nums) == 8, (file_txt, cell_ln)
            row = eval(cell_nums[0])
            col = eval(cell_nums[1])
            if binary_label:
                label = ''.join(cell_nums[2:])
            else:
                label = ''
                for i, c in enumerate(cell_nums[2:]):
                    if c == '1':
                        label += str(i+1)
                    else:
                        assert c == '0', (file_txt, cell_ln, i, c)
            left = v_lines[(col-1)*2]
            right = v_lines[(col-1)*2+1]
            top = h_lines[(row-1)*3]
            bottom = h_lines[(row-1)*3+2]
            cells.append(CellInfo(row=row, col=col,
                                  left=left, top=top, right=right, bottom=bottom,
                                  label=label))
    return angle, h_lines, v_lines, cells