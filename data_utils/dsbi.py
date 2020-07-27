#!/usr/bin/env python
# coding: utf-8
"""
Utils for DSBI dataset (https://github.com/yeluo1994/DSBI)
"""
import collections
from braille_utils import label_tools as lt

CellInfo = collections.namedtuple('CellInfo', 
                                  ['row', 'col',  # row and column in a symbol grid
                                   'left', 'top', 'right', 'bottom',  # symbol corner coordinates in pixels
                                   'label'])  # symbol label either like '246' or '010101' format

def read_txt(file_txt, binary_label = True):
    """
    Loads Braille annotation from DSBI annotation txt file
    :param file_txt: filename of txt file
    :param binary_label: return symbol label in binary format, like '010101' (if True),
        or human readable like '246' (if False)
    :return: tuple (
        angle: value from 1st line of annotation file,
        h_lines: list of horizontal lines Y-coordinates,
        v_lines: list of vertical lines X-coordinates,,
        cells: symbols as list of CellInfo
    )
    None, None, None, None for empty annotation
    """
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


def read_DSBI_annotation(label_filename, width, height, rect_margin, get_points):
    """
    Loads Braille annotation from DSBI annotation txt file in albumentations format
    :param label_filename: filename of txt file
    :param width: image width
    :param height: image height
    :param rect_margin:
    :param get_points: Points or Symbols mode
    :return:
        List of symbol rects if get_points==False. Each rect is a tuple (left, top, right, bottom, label) where
        left..bottom are in [0,1], label is int in [1..63]. Symbol size is extended to rect_margin*width of symbol
        in every side.
        List of points rects if get_points==True. Each point is a tuple (left, top, right, bottom, label) where
        left..bottom are in [0,1], label is 0. Width and height of point is 2*rect_margin*width of symbol
    """
    _, _, _, cells = read_txt(label_filename, binary_label=True)
    if cells is not None:
        if get_points:
            rects = []
            for cl in cells:
                w = int((cl.right - cl.left) * rect_margin)
                h = w
                for i in range(6):
                    if cl.label[i] == '1':
                        iy = i % 3
                        ix = i - iy
                        if ix == 0:
                            xc = cl.left
                        else:
                            xc = cl.right
                        left, right = xc - w, xc + w
                        if iy == 0:
                            yc = cl.top
                        elif iy == 1:
                            yc = (cl.top + cl.bottom) // 2
                        else:
                            yc = cl.bottom
                        top, bottom = yc - h, yc + h
                        rects.append([left / width, top / height, right / width, bottom / height, 0])
        else:
            rects = [(
                (c.left - rect_margin * (c.right - c.left)) / width,
                (c.top - rect_margin * (c.right - c.left)) / height,
                (c.right + rect_margin * (c.right - c.left)) / width,
                (c.bottom + rect_margin * (c.right - c.left)) / height,
                lt.label010_to_int(c.label),
                ) for c in cells if c.label != '000000']
    else:
        rects = []
    return rects