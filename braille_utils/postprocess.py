from functools import cmp_to_key
from collections import defaultdict

import braille_utils.letters as letters
import braille_utils.label_tools as lt


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
    LINE_THR = 0.8
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
            if math_mode and ch.char is None:
                frac_mode = False
                if math_lang:
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, [math_lang.upper()])
                    if ch.char is not None:
                        if math_lang.isupper():
                            ch.char = ch.char.upper()
                    else:
                        math_lang = ''
                if not math_lang and (ch.spaces_before or lt.int_to_letter(ch.label, ['MATH_RU']) == '..'):
                    # без spaces_before точка и запятая после числа интерпретируется как математический знак :
                    # перед умножением точкой (..) пробел не ставится
                    ch.labeling_char = ch.char = lt.int_to_letter(ch.label, ['MATH_RU'])
                    if ch.char is not None:
                        if ch.char in {'en','EN'}:
                            math_lang = ch.char
                            ch.char = ''
                        elif ch.char == '..':
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
                if prev_ch and prev_ch.char == ",":
                    prev_ch.char = ", "
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
    'EN': interpret_line_RU, # TODO in can work with some errors for EN
}


def boxes_to_lines(boxes, labels, lang):
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


if __name__ == '__main__':
    #OK
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

