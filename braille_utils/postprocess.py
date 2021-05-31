import cv2
from collections import defaultdict
from functools import cmp_to_key
import numpy as np
import PIL

from braille_utils import letters
from braille_utils import label_tools as lt


class LineChar:
    def __init__(self, box, label):
        self.original_box = box # box found by NN
        self.x = (box[0] + box[2])/2 # original x of last char
        self.y = (box[1] + box[3])/2 # original y of last char
        self.w = (box[2]-box[0]) # original w
        self.h = (box[3]-box[1]) # original h
        self.approximation = None # approximation (err,a,b,w,h) on interval ending on this
        self.refined_box = box # refined box
        self.label = label
        self.spaces_before = 0
        self.char = '' # char to display in printed text
        self.labeling_char = '' # char to display in rects labeling


class Line:
    STEP_TO_W = 1.25
    LINE_THR = 0.6
    AVG_PERIOD = 5 # для аппроксимации при коррекции
    AVG_APPROX_DIST = 3 # берутся точки с интервалос не менее 2, т.е. 0я и 3я или 1я и 4я

    def __init__(self, box, label):
        self.chars = []
        new_char = LineChar(box, label)
        self.chars.append(new_char)
        self.x = new_char.x
        self.y = new_char.y
        self.h = new_char.h
        self.slip = 0
        self.has_space_before = False

    def check_and_append(self, box, label):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        if abs(self.y + self.slip * (x-self.x) -y) < self.h*self.LINE_THR:
            new_char = LineChar(box, label)
            self.chars.append(new_char)
            calc_chars = self.chars[-self.AVG_PERIOD:]
            new_char.approximation = self._calc_approximation(calc_chars)
            if new_char.approximation is None:
                self.x = new_char.x
                self.y = new_char.y
                self.h = new_char.h
            else:
                err, a, b, w, h = new_char.approximation
                self.x = new_char.x
                self.y = self.x * a + b
                self.h = h
                self.slip = a
            return True
        return False

    def _calc_approximation(self, calc_chars):
        if len(calc_chars) <= self.AVG_APPROX_DIST:
            return None
        best_pair = [1e99, None, None, None, None, ] # err,i,j,k,a,b
        for i in range(len(calc_chars)-self.AVG_APPROX_DIST):
            for j in range(i+self.AVG_APPROX_DIST, len(calc_chars)):
                a = (calc_chars[j].y - calc_chars[i].y) / (calc_chars[j].x - calc_chars[i].x)
                b = (calc_chars[j].x * calc_chars[i].y - calc_chars[i].x * calc_chars[j].y) / (calc_chars[j].x - calc_chars[i].x)
                ks = [k for k in range(len(calc_chars)) if k!=i and k!= j]
                errors = [abs(calc_chars[k].x * a + b - calc_chars[k].y) for k in ks]
                min_err = min(errors)
                if min_err < best_pair[0]:
                    idx_if_min = min(range(len(errors)), key=errors.__getitem__)
                    best_pair = min_err, i, j, ks[idx_if_min], a, b
        err, i, j, k, a, b = best_pair
        w = (calc_chars[i].w + calc_chars[j].w + calc_chars[k].w) / 3
        h = (calc_chars[i].h + calc_chars[j].h + calc_chars[k].h) / 3
        return err, a, b, w, h

    def refine(self):
        for i in range(len(self.chars)):
            curr_char = self.chars[i]
            ref_chars = [ch for ch in  self.chars[i:i+self.AVG_PERIOD] if ch.approximation is not None]
            if ref_chars:
                best_char = min(ref_chars, key = lambda ch: ch.approximation[0])
                err, a, b, w, h = best_char.approximation
                expected_x = curr_char.x
                expected_y = expected_x * a + b
                curr_char.refined_box = [expected_x-w/2, expected_y-h/2, expected_x+w/2, expected_y+h/2]
            if i > 0:
                step = (curr_char.refined_box[2] - curr_char.refined_box[0]) * self.STEP_TO_W
                prev_char = self.chars[i-1]
                curr_char.spaces_before = max(0, round(0.5 * ((curr_char.refined_box[0] + curr_char.refined_box[2]) - (prev_char.refined_box[0] + prev_char.refined_box[2])) / step) - 1)


def get_compareble_y(line1, line2):
    line1, line2, sign = (line1, line2, 1) if line1.length > line2.length else (line2, line1, -1)
    if line2.chars[0].x > line1.middle_x:
        y1 = line1.chars[-1].y + line1.mean_slip * (line2.chars[0].x - line1.chars[-1].x)
    else:
        y1 = line1.chars[0].y + line1.mean_slip * (line2.chars[0].x - line1.chars[0].x)
    if sign > 0:
        return y1, line2.chars[0].y
    else:
        return line2.chars[0].y, y1


def _sort_lines(lines):

    def _cmp_lines(line1, line2):
        y1, y2 = get_compareble_y(line1, line2)
        return 1 if y1 > y2 else -1

    for ln in lines:
        ln.length = ln.chars[-1].x - ln.chars[0].x
        ln.middle_x = 0.5*(ln.chars[-1].x + ln.chars[0].x)
        if ln.length > 0:
            ln.mean_slip = (ln.chars[-1].y - ln.chars[0].y)/ln.length
        else:
            ln.mean_slip = 0
    return sorted(lines, key = cmp_to_key(_cmp_lines))


def interpret_line_RU(line, lang, mode = None):
    '''
    precess line of chars and fills char and labeling_char attributes of chars according to language rules
    :param line: list of LineChar. LineChar must have spaces_before, char and labeling_char attributes
    :param lang: 'RU' etc.
    :return: None
    '''
    if mode is None:
        mode = defaultdict(bool)
    else:
        mode = defaultdict(bool, mode)
    digit_mode = mode['digit_mode']
    frac_mode = mode['frac_mode']
    math_mode = mode['math_mode']
    math_lang = mode['math_lang']
    if not math_lang:
        math_lang = ''
    caps_mode = mode['caps_mode']
    brackets_on = mode['brackets_on']
    if not brackets_on:
        brackets_on = defaultdict(int)

    prev_ch = None
    for i, ch in enumerate(line.chars):
        ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['SYM'])
        if ch.char == letters.markout_sign:
            ch.char = ''
            if i < len(line.chars)-1:
                line.chars[i+1].spaces_before += ch.spaces_before
                ch.spaces_before = 0
        elif ch.char == letters.num_sign:
            ch.char = ''
            if digit_mode:
                ch.spaces_before += 1 # need to separate from previous number or separate fraction part from integer part
            digit_mode = True
            math_mode = True
            if prev_ch is not None and prev_ch.labeling_char == 'н':
                prev_ch.char = '№'
        else:
            if ch.spaces_before:
                digit_mode = False
                frac_mode = False
            ch.char = None
            if digit_mode:
                if not frac_mode:
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['NUM'])
                    if ch.char is None:
                        ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['NUM_DENOMINATOR'])
                        if ch.char is not None:
                            if ch.char == '/0' and prev_ch.labeling_char == '0':
                                prev_ch.char = ''
                                ch.char = '%'
                            else:
                                ch.labeling_char = ch.char = None #frac_mode = True # TOOD вернуться к отображению дробей
                else:
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['NUM_DENOMINATOR'])
                    if ch.char is not None:
                        ch.char = ch.char[1:]
            if ch.char is None:
                if ch.spaces_before:
                    math_lang = ''
                if math_lang:
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, [math_lang.upper()])
                    if ch.char is not None:
                        if math_lang.isupper():
                            ch.char = ch.char.upper()
                        digit_mode = False
                        if not ch.char.isalpha():
                            math_lang = ''
                    else:
                        math_lang = ''
            if ch.char is None:
                if (ch.spaces_before
                    or not prev_ch
                    or prev_ch and not prev_ch.char.isalpha()
                    or math_mode and not math_lang and lt.int_to_letter(ch.label, ['MATH_RU']) == '..'
                ):
                    if lt.int_to_letter(ch.label, ['MATH_RU']) in {'en', 'EN'}:
                        math_lang = lt.int_to_letter(ch.label, ['MATH_RU'])
                        ch.char = ''
            if math_mode and ch.char is None:
                frac_mode = False
                if not math_lang and (ch.spaces_before or lt.int_to_letter(ch.label, ['MATH_RU']) == '..'):
                    # без spaces_before точка и запятая после числа интерпретируется как математический знак :
                    # перед умножением точкой (..) пробел не ставится
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['MATH_RU'])
                    if ch.char is not None:
                        if ch.char == '..':
                            if i < len(line.chars)-1 and line.chars[i+1].spaces_before == 0 and lt.int_to_letter(line.chars[i+1].label, ['NUM']) is not None:
                                ch.char = '.'
                            else:
                                ch.char = '*'
                        elif ch.char == '::':
                            ch.char = ':'
                        ch.spaces_before = max(0, ch.spaces_before-1)
            if ch.char is None:
                ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['SYM', lang])
                if ch.char not in {',', '.', '(', ')'}:
                    math_mode = False
                    math_lang = ''
                    digit_mode = False
                    frac_mode = False
            if not math_mode:
                if prev_ch and prev_ch.char in {",", ";"}:
                    prev_ch.char += " "
            if ch.char == '()':
                if brackets_on['(('] == 0:
                    ch.char = '('
                    brackets_on['(('] = 1
                else:
                    ch.char = ')'
                    brackets_on['(('] = 0
            elif ch.char == 'ъ' and (ch.spaces_before or prev_ch is None or not prev_ch.char.isalpha()):
                ch.char = '['
                brackets_on['['] += 1
            elif ch.char == 'ь' and i < len(line.chars)-1 and (line.chars[i+1].spaces_before or True # TODO
                                                                ) and brackets_on['['] > 0:
                ch.char = ']'
                brackets_on['['] -= 1
            if ch.char is None:
                ch.labeling_char = '~' + lt.int_to_label123(ch.label)
                ch.char = ch.labeling_char + '~'
            if caps_mode:
                ch.char = ch.char.upper()
                caps_mode = False
            if ch.char == letters.caps_sign:
                caps_mode = True
                ch.char = ''
            if ch.char == 'EN':
                caps_mode = True
                ch.char = ''  # TODO
        prev_ch = ch

    return {
        #'digit_mode': digit_mode,
        #'frac_mode': frac_mode,
        #'math_mode': math_mode,
        #'math_lang': math_lang,
        #'caps_mode': caps_mode,
        'brackets_on': brackets_on,
    }


interpret_line_funcs = {
    'RU': interpret_line_RU,
    'UZ': interpret_line_RU,
    'UZL': interpret_line_RU,
    'GR': interpret_line_RU,
    'LV': interpret_line_RU,
    'EN': interpret_line_RU, # TODO in can work with some errors for EN
}


def filter_lonely_rects_for_lines(lines):
    allowed_lonely = {} # lt.label010_to_int('111000'), lt.label010_to_int('000111'), lt.label010_to_int('111111')
    filtered_chars = []
    for ln in lines:
        while len(ln.chars) and (ln.chars[0].label not in allowed_lonely and len(ln.chars)>1 and ln.chars[1].spaces_before > 1 or len(ln.chars) == 1):
            filtered_chars.append(ln.chars[0])
            ln.chars = ln.chars[1:]
            if len(ln.chars):
                ln.chars[0].spaces_before = 0
        while len(ln.chars) and (ln.chars[-1].label not in allowed_lonely and len(ln.chars)>1 and ln.chars[-1].spaces_before > 1 or len(ln.chars) == 1):
            filtered_chars.append(ln.chars[-1])
            ln.chars = ln.chars[:-1]
    return [ln for ln in lines if len(ln.chars)], filtered_chars


def boxes_to_lines(boxes, labels, lang, filter_lonely = True):
    '''
    :param boxes: list of (left, tor, right, bottom)
    :return: text: list of strings
    '''
    VERTICAL_SPACING_THR = 2.3

    boxes = list(zip(boxes, labels))
    lines = []
    boxes = sorted(boxes, key=lambda b: b[0][0])
    for b in boxes:
        found_line = None
        for ln in lines:
            if ln.check_and_append(box=b[0], label=b[1]):
                # to handle seldom cases when one char can be related to several lines mostly because of errorneous outlined symbols
                if (found_line and (found_line.chars[-1].x - found_line.chars[-2].x) < (ln.chars[-1].x - ln.chars[-2].x)):
                    ln.chars.pop()
                else:
                    if found_line:
                        found_line.chars.pop()
                    found_line = ln
        if found_line is None:
            lines.append(Line(box=b[0], label=b[1]))

    lines = _sort_lines(lines)
    interpret_line_f = interpret_line_funcs[lang]
    interpret_line_mode = None
    prev_line = None
    for ln in lines:
        ln.refine()

        if prev_line is not None:
            prev_y, y = get_compareble_y(prev_line, ln)
            if (y - prev_y) > VERTICAL_SPACING_THR * ln.h:
                ln.has_space_before = True
        prev_line = ln

        interpret_line_mode = interpret_line_f(ln, lang, mode = interpret_line_mode)
    if filter_lonely:
        lines, _ = filter_lonely_rects_for_lines(lines)
    return lines


def string_to_line(text_line):
    '''
    :param text_line: single line text
    :return: Line (not postprocessed)
    '''
    ext_mode = False
    line = None
    spaces_before = 0
    for ch in text_line:
        if ch == '~':
            if not ext_mode:
                ext_mode = True
                ext_ch = ''
                ch = None
            else:
                ext_mode = False
                if ext_ch.isdigit():
                    ext_ch = '~'+ext_ch
                ch = ext_ch
        if ch:
            if ext_mode:
                ext_ch += ch
            else:
                if ch == ' ':
                    spaces_before += 1
                else:
                    label = lt.human_label_to_int(ch)
                    if line is None:
                        line = Line(box=[0,0,0,0], label=label)
                    else:
                        line.chars.append(LineChar(box=[0,0,0,0], label=label))
                    line.chars[-1].spaces_before = spaces_before
                    spaces_before = 0
    assert not ext_mode, text_line
    return line


def text_to_lines(text, lang = 'RU'):
    '''
    :param text: multiline text
    :return: list of Line
    '''
    text_lines = text.splitlines()
    interpret_line_mode = None
    has_space_before = False
    lines = []
    for tln in text_lines:
        if not tln.strip():
            has_space_before = True
        else:
            ln = string_to_line(tln)
            ln.has_space_before = has_space_before
            has_space_before = False
            lines.append(ln)
    interpret_line_f = interpret_line_funcs[lang]
    interpret_line_mode = None
    for ln in lines:
        interpret_line_mode = interpret_line_f(ln, lang, mode = interpret_line_mode)
    return lines


def lines_to_text(lines):
    '''
    :param lines: list of Line (returned by boxes_to_lines or text_to_lines)
    :return: text as string
    '''
    out_text = []
    for ln in lines:
        if ln.has_space_before:
            out_text.append('')
        s = ''
        for ch in ln.chars:
            s += ' ' * ch.spaces_before + ch.char
        out_text.append(s)
    return '\n'.join(out_text)


def validate_postprocess(in_text, out_text):
    '''
    :return: validates that  in_text -> out_text
    '''
    res_text = lines_to_text(text_to_lines(in_text))
    assert res_text == out_text, (in_text, res_text, out_text)


#####################################
# find transformation
#####################################

MIN_RECTS = 10
MAX_ROTATION = 0.2
MIN_ROTATION = 0.02

def center_of_char(ch):
    """
    :return: x,y - center of ch.refined_box
    """
    return (ch.refined_box[0] + ch.refined_box[2])/2, (ch.refined_box[1] + ch.refined_box[3])/2

def find_line(ch1, ch2):
    """
    defines line than crossing centers of 2 chars: ax + by = 1
    :param ch1:
    :param ch2:
    :return: a,b
    """
    x1, y1 = center_of_char(ch1)
    x2, y2 = center_of_char(ch2)
    d = x1*y2 - x2*y1
    return (y2 - y1)/d, (x1 - x2)/d


def find_cross(ln1, ln2):
    """
    finds cross points of 2 lines. Lines are defined as (a,b): ax+by=1
    :param ln1:
    :param ln2:
    :return: (x,y)
    """
    d = ln1[0]*ln2[1] - ln2[0]*ln1[1]
    return (ln2[1]-ln1[1])/d, (ln1[0]-ln2[0])/d


def calc_v_err(ch, line):
    """
    calculates error between ch and horizontal line. If ch is not refered to line, return None
    :param ch:
    :param line:
    :return: err
    """
    w = ch.refined_box[2] - ch.refined_box[0]
    thr = w*0.25
    x, y = center_of_char(ch)
    yv = (1 - line[0]*x) / line[1]
    if abs(yv - y) > thr:
        return 1.
    else:
        return abs(yv - y) / w


def calc_h_err(ch, line):
    """
    calculates error between ch and vertical line. If ch is not refered to line, return None
    :param ch:
    :param line:
    :return: err
    """
    w = ch.refined_box[2] - ch.refined_box[0]
    thr = w*0.25
    x, y = center_of_char(ch)
    xv = (1 - line[1]*y) / line[0]
    if xv < ch.refined_box[0] or xv > ch.refined_box[2]:
        return None
    if abs(xv - x) > thr:
        return 1.
    else:
        return abs(xv - x) / w



def find_best_h_line(chars, bounds):
    """
    finds best horisontal line for line of chars
    :param chars:
    :param bounds:
    :return: best_err, best_line
    """
    best_err = None
    best_line = None
    if len(chars) >= MIN_RECTS:
        w = bounds[2] - bounds[0]
        for left_ch in chars:
            if left_ch.refined_box[0] > bounds[0] + w / 3:
                break
            for right_ch in reversed(chars):
                if right_ch.refined_box[2] < bounds[2] - w / 3:
                    break
                ln = find_line(left_ch, right_ch)
                err = 0
                n = 0
                for ch in chars:
                    err_i = calc_v_err(ch, ln)
                    if err_i is not None:
                        err += err_i
                        n += 1
                if n >= MIN_RECTS:
                    err /= n
                    if best_err is None or err < best_err:
                        best_err = err
                        best_line = ln
    return best_err, best_line


def find_best_v_lines(top_ln, bot_ln, lines, bounds):
    """
    finds best 2 vertical lines for lines of chars
    :return:
    """
    best_err = [None] * 2
    best_line = [None] * 2
    if len(lines) >= MIN_RECTS:
        w = bounds[2] - bounds[0]
        location = 0  # left or right side
        for top_ch in top_ln.chars:
            if top_ch.refined_box[0] >  bounds[0] + w / 3:
                if top_ch.refined_box[2] <  bounds[2] - w / 3:
                    continue
                else:
                    location = 1
            topx, topy = center_of_char(top_ch)
            bot_gen = bot_ln.chars if location == 0 else reversed(bot_ln.chars)
            for bottom_ch in bot_gen:
                botx, boty = center_of_char(bottom_ch)
                if location == 0:
                    if botx < topx - MAX_ROTATION * (boty - topy):
                        continue
                    if botx > topx + MAX_ROTATION * (boty - topy):
                        break
                else:
                    if botx > topx + MAX_ROTATION * (boty - topy):
                        continue
                    if botx < topx - MAX_ROTATION * (boty - topy):
                        break
                ln = find_line(top_ch, bottom_ch)
                err = 0
                n = 0
                for i_ln in lines:
                    for ch in i_ln.chars:
                        err_i = calc_h_err(ch, ln)
                        if err_i is not None:
                            err += err_i
                            n += 1
                    if n >= MIN_RECTS:
                        err /= n
                        if best_err[location] is None or err < best_err[location]:
                            best_err[location] = err
                            best_line[location] = ln
    return best_err[0], best_line[0], best_err[1], best_line[1]


def find_transformation_full(lines):
    """
    Finds alignment transform. Very slow and distortionable
    :param lines:
    :return:
    """
    if len(lines) == 0:
        return
    bounds = lines[0].chars[0].refined_box
    for ln in lines:
        for ch in ln.chars:
            bounds = [min(bounds[0], ch.refined_box[0]), min(bounds[1], ch.refined_box[1]),
                      max(bounds[2], ch.refined_box[2]), max(bounds[3], ch.refined_box[3])]
    best_lines = [None] * 4  # left, top, right, bottom lines
    best_errors = [None] * 4  # errors for best_lines
    bot_lines = len(lines) // 4 if len(lines) >= MIN_RECTS else 0
    top_lines = len(lines) // 4 if bot_lines > 0 else len(lines)
    for top_i in range(top_lines):
        top_ln = lines[top_i]
        err, ln = find_best_h_line(top_ln.chars, bounds)
        if err is not None:
            if best_errors[1] is None or err < best_errors[1]:
                best_errors[1], best_lines[1] = err, ln
        for bot_i in range(len(lines) - bot_lines, len(lines)):
            bot_ln = lines[bot_i]
            err, ln = find_best_h_line(bot_ln.chars, bounds)
            if err is not None:
                if best_errors[3] is None or err < best_errors[3]:
                    best_errors[3], best_lines[3] = err, ln
            err1, ln1, err2, ln2 = find_best_v_lines(top_ln, bot_ln, lines, bounds)
            if err1 is not None:
                if best_errors[0] is None or err1 < best_errors[0]:
                    best_errors[0], best_lines[0] = err1, ln1
            if err2 is not None:
                if best_errors[2] is None or err2 < best_errors[2]:
                    best_errors[2], best_lines[2] = err2, ln2
    print(best_errors)
    if best_errors[0] is not None and best_errors[1] is not None and best_errors[2] is not None and best_errors[3] is not None:
        src_points = np.array([find_cross(best_lines[0], best_lines[1]), find_cross(best_lines[2], best_lines[1]),   # 0 1
                               find_cross(best_lines[0], best_lines[3]), find_cross(best_lines[2], best_lines[3])])  # 2 3
        src_bounds = [min(src_points[0,0], src_points[2,0]), min(src_points[0,1], src_points[1,1]),
                      max(src_points[1,0], src_points[3,0]), max(src_points[2,1], src_points[3,1]),]  # left top r. bot.
        target_points = np.array([(src_bounds[0], src_bounds[1]), (src_bounds[2], src_bounds[1]),
                                  (src_bounds[0], src_bounds[3]), (src_bounds[2], src_bounds[3])])
        hom, mask = cv2.findHomography(src_points, target_points)
    elif best_errors[1] is not None:
        angle = -best_lines[1][0]/best_lines[1][1]*59  # rad -> grad
        hom = cv2.getRotationMatrix2D(((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2), angle, 1.)
    else:
        hom = None
    return hom


def find_transformation(lines, img_size_wh):
    """
    Finds alignment transform
    :param lines:
    :return:
    """
    hom = None
    if len(lines) > 0:
        bounds = lines[0].chars[0].refined_box
        for ln in lines:
            for ch in ln.chars:
                bounds = [min(bounds[0], ch.refined_box[0]), min(bounds[1], ch.refined_box[1]),
                          max(bounds[2], ch.refined_box[2]), max(bounds[3], ch.refined_box[3])]
        sum_slip = 0
        sum_weights = 0
        for ln in lines:
            if len(ln.chars) > MIN_RECTS:
                pt1 = center_of_char(ln.chars[len(ln.chars)//5])
                pt2 = center_of_char(ln.chars[len(ln.chars)*4//5])
                dx = pt2[0] - pt1[0]
                slip = (pt2[1] - pt1[1])/dx
                sum_slip += slip * dx
                sum_weights += dx
        if sum_weights > 0:
            angle = sum_slip / sum_weights  # rad -> grad
            if abs(angle) < MAX_ROTATION and abs(angle) > MIN_ROTATION:
                scale = 1.
                center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
                hom = cv2.getRotationMatrix2D(center, angle * 59, scale)
                old_points = np.array([[(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[0], bounds[3]), (bounds[2], bounds[3])]])
                new_points = cv2.transform(old_points, hom)
                scale = min(
                    center[0]/(center[0]-new_points[0][0][0]), center[1]/(center[1]-new_points[0][0][1]),
                    (img_size_wh[0]-center[0]) / (new_points[0][1][0] - center[0]), center[1]/(center[1]-new_points[0][1][1]),
                    center[0] / (center[0] - new_points[0][2][0]), (img_size_wh[1]-center[1]) / (new_points[0][2][1] - center[1]),
                    (img_size_wh[0]-center[0]) / (new_points[0][3][0] - center[0]), (img_size_wh[1]-center[1]) / (new_points[0][3][1] - center[1])
                )
                if scale < 1:
                    hom = cv2.getRotationMatrix2D(center, angle * 59, scale)
    return hom


def transform_image(img, hom):
    """
    transforms img and refined_box'es and original_box'es for chars at lines using homography matrix found by
    find_transformation(lines)
    :param img: PIL image
    :param lines:
    :param hom:
    :return: img, lines transformed
    """
    if hom.shape[0] == 3:
        img_transform = cv2.warpPerspective
    else:
        img_transform = cv2.warpAffine
    img = img_transform(np.asarray(img), hom, img.size, flags=cv2.INTER_LINEAR )
    img = PIL.Image.fromarray(img)
    return img

def transform_lines(lines, hom):
    if hom.shape[0] == 3:
        pts_transform = cv2.perspectiveTransform
    else:
        pts_transform = cv2.transform
    old_centers = np.array([[center_of_char(ch) for ln in lines for ch in ln.chars]])
    new_centers = pts_transform(old_centers, hom)
    shifts = (new_centers - old_centers)[0]
    i = 0
    for ln in lines:
        for ch in ln.chars:
            ch.refined_box[0] += shifts[i, 0]
            ch.refined_box[1] += shifts[i, 1]
            ch.refined_box[2] += shifts[i, 0]
            ch.refined_box[3] += shifts[i, 1]
            ch.original_box[0] += shifts[i, 0]
            ch.original_box[1] += shifts[i, 1]
            ch.original_box[2] += shifts[i, 0]
            ch.original_box[3] += shifts[i, 1]
            i += 1
    return lines

def transform_rects(rects, hom):
    if len(rects):
        if hom.shape[0] == 3:
            pts_transform = cv2.perspectiveTransform
        else:
            pts_transform = cv2.transform
        old_centers = np.array([[((r[2]+r[0])/2, (r[3]+r[1])/2) for r in rects]])
        new_centers = pts_transform(old_centers, hom)
        shifts = (new_centers - old_centers)[0].tolist()
        rects = [
            (x[0][0] + x[1][0], x[0][1] + x[1][1], x[0][2] + x[1][0], x[0][3] + x[1][1]) + tuple(x[0][4:])
            for x in zip(rects, shifts)
        ]
    return rects


if __name__ == '__main__':
    #OK
    validate_postprocess('''аб«~6~и»вг''', '''аб«i»вг''')
    validate_postprocess('''~46~и вг''', '''I вг''')
    validate_postprocess('''~##~2))~6~r9n7o''', '''2))ringo''')

    validate_postprocess('''(~##~1) =~##~1''', '''(1)=1''')
    validate_postprocess('''а ~((~б~))~,''', '''а (б),''')
    validate_postprocess('''~((~в~))~,''', '''(в),''')
    validate_postprocess('''~()~~##~1~()~,''', '''(1),''')
    validate_postprocess('''~##~1,ма,''', '''1, ма,''')
    validate_postprocess('''~##~20-х годах''', '''20-х годах''')
    validate_postprocess('''~##~1\n0''', '''1\nж''')

    validate_postprocess('''ab  c

d e f''', '''аб  ц

д е ф''')

    validate_postprocess('''~1~b  c~##~34

d e f''', '''аб  ц34

д е ф''')

