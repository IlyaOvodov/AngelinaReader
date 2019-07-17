from functools import cmp_to_key

import braille_utils.letters as letters
import braille_utils.label_tools as lt


class LineChar:
    def __init__(self, box, label):
        box = [b.item() for b in box]
        self.original_box = box # box found by NN
        self.x = (box[0] + box[2])/2 # original x of last char
        self.y = (box[1] + box[3])/2 # original y of last char
        self.w = (box[2]-box[0]) # original w
        self.h = (box[3]-box[1]) # original h
        self.approximation = None # approximation (err,a,b,w,h) on interval ending on this
        self.refined_box = box # refined box
        self.label = label
        self.spaces_before = 0
        self.char = ''


class Line:
    STEP_TO_W = 1.25
    LINE_THR = 0.5
    AVG_PERIOD = 5 # для аппроксимации при коррекции
    AVG_APPROX_DIST = 3 # берутся точки с интервалос не менее 2, т.е. 0я и 3я или 1я и 4я

    def __init__(self, box, label):
        self.chars = []
        new_char = LineChar(box, label)
        self.chars.append(new_char)
        self.x = new_char.x
        self.y = new_char.y
        self.h = new_char.h
        self.slip = None

    def check_and_append(self, box, label):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        if abs(self.y + (self.slip if self.slip is not None else 0) *(x-self.x) -y) < self.h*self.LINE_THR:
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


def _sort_lines(lines):

    def _cmp_lines(line1, line2):
        line1, line2, sign = (line1, line2, 1) if line1.length > line2.length else (line2, line1, -1)
        if line2.chars[0].x > line1.middle_x:
            return sign if line1.chars[-1].y + line1.mean_slip * (line2.chars[0].x - line1.chars[-1].x) > line2.chars[0].y else -sign
        else:
            return sign if line1.chars[0].y  + line1.mean_slip * (line2.chars[0].x - line1.chars[0].x ) > line2.chars[0].y else -sign

    for ln in lines:
        ln.length = ln.chars[-1].x - ln.chars[0].x
        ln.middle_x = 0.5*(ln.chars[-1].x + ln.chars[0].x)
        if ln.length > 0:
            ln.mean_slip = (ln.chars[-1].y - ln.chars[0].y)/ln.length
        else:
            ln.mean_slip = 0
    return sorted(lines, key = cmp_to_key(_cmp_lines))


def boxes_to_lines(boxes, labels, lang = 'RU'):
    '''
    :param boxes: list of (left, tor, right, bottom)
    :return: text: list of strings
    '''
    boxes = list(zip(boxes, labels))
    lines = []
    boxes = sorted(boxes, key=lambda b: b[0][0])
    for b in boxes:
        for l in lines:
            if l.check_and_append(box=b[0], label=b[1]):
                break
        else:
            lines.append(Line(box=b[0], label=b[1]))

    lines = _sort_lines(lines)
    for l in lines:
        l.refine()
        digit_mode = False
        caps_mode = False
        for ch in l.chars:
            if ch.spaces_before:
                digit_mode = False
                caps_mode = False
            lbl = lt.int_to_letter(ch.label.item(), 'NUM' if digit_mode else lang)
            if lbl == letters.markout_sign:
                ch.char = ''
            elif lbl == letters.num_sign:
                digit_mode = True
                ch.char = ''
            elif lbl == letters.caps_sign:
                caps_mode = True
                ch.char = ''
            else:
                if not lbl:
                    digit_mode = False
                    lbl = lt.int_to_label123(ch.label.item())
                    lbl = '&'+lbl+'&'
                if caps_mode:
                    lbl = lbl.upper()
                    caps_mode = False
                ch.char = lbl
    return lines



if __name__ == '__main__':
    pass
