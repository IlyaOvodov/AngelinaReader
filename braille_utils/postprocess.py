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
        self.refined_box = box # refined box
        self.label = label
        self.spaces_before = 0
        self.char = ''


class Line:
    SPACE_TO_H = 1.0 # 6.6/(2.7*2+1.5) = 0.96, 6.0/(2.5*2+1.3)=0.95, 6.5/(2.5*2+1.3) = 1.03
    LINE_THR = 0.5
    AVG_PERIOD = 5
    def __init__(self, box, label):
        self.chars = []
        self.slip = 0
        self._append_char(box, label)

    def _append_char(self, box, label):
        new_char = LineChar(box, label)
        self.chars.append(new_char)
        self.x = new_char.x
        self.y = new_char.y
        self.h = new_char.h
        return new_char

    def _calc_err(self, chars, i, j, k):
        factor = (chars[k].x - chars[i].x)/(chars[j].x - chars[i].x)
        y = chars[i].y + (chars[j].y - chars[i].y) * factor
        d = min(abs(chars[k].x - chars[i].x), abs(chars[k].x - chars[j].x))
        d2 = min(abs(chars[self.AVG_PERIOD-1].x - chars[i].x), abs(chars[-self.AVG_PERIOD].x - chars[j].x))
        assert d != 0
        return abs(y-chars[k].y)*d2/d

    def check_and_append(self, box, label):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        if abs(self.y + self.slip*(x-self.x) -y) < self.h*self.LINE_THR:
            new_char = self._append_char(box, label)
            if len(self.chars) >= self.AVG_PERIOD:
                calc_chars = self.chars[-self.AVG_PERIOD:]
                best_pair = None
                best_err = 1e99
                for i in range(self.AVG_PERIOD-1):
                    for j in range(i+1, self.AVG_PERIOD):
                        err = min([self._calc_err(calc_chars, i, j, k) for k in range(self.AVG_PERIOD) if k!=i and k!= j])
                        if err < best_err:
                            best_err, best_pair = err, (i,j)
                i, j = best_pair
                chars_before = sum([ch.spaces_before for ch in calc_chars[i+1: j+1]]) + j - i
                step = (calc_chars[j].x-calc_chars[i].x)/chars_before
                ref_x = (calc_chars[j].x + calc_chars[i].x + chars_before*step)/2
                dist_in_chars = round((new_char.x - ref_x) / step)
                new_char.spaces_before = max(0, dist_in_chars - 1 - (self.AVG_PERIOD - 1 - j))
                expected_x = ref_x + step*dist_in_chars
                factor = (expected_x - calc_chars[i].x) / (calc_chars[j].x - calc_chars[i].x)
                expected_y = calc_chars[i].y + (calc_chars[j].y - calc_chars[i].y) * factor
                w, h = (calc_chars[i].w + calc_chars[j].w)/2, (calc_chars[i].h + calc_chars[j].h)/2
                new_char.refined_box = [expected_x-w/2, expected_y-h/2, expected_x+w/2, expected_y+h/2]
                self.x, self.y = expected_x, expected_y
                self.h = h
                self.slip = (calc_chars[j].y - calc_chars[i].y)/(calc_chars[j].x - calc_chars[i].x)
            elif len(self.chars) >= 2:
                step = new_char.h * self.SPACE_TO_H
                new_char.spaces_before = max(0, round((new_char.x - self.chars[-2].x)/step)-1)
            return True
        return False


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

    lines = sorted(lines, key = lambda l: l.y)
    for l in lines:
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
