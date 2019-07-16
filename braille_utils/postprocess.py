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
    SPACE_TO_H = 0.95 # 6.6/(2.7*2+1.5) = 0.96, 6.0/(2.5*2+1.3)=0.95, 6.5/(2.5*2+1.3) = 1.03
    LINE_THR = 0.5
    AVG_PERIOD = 5
    def __init__(self, box, label):
        self.chars = []
        new_char = LineChar(box, label)
        self.chars.append(new_char)
        self.x = new_char.x
        self.y = new_char.y
        self.h = new_char.h
        self.slip = 0

    def check_and_append(self, box, label):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        if abs(self.y + self.slip*(x-self.x) -y) < self.h*self.LINE_THR:
            new_char = LineChar(box, label)
            if len(self.chars) >= self.AVG_PERIOD:
                calc_chars = self.chars[-self.AVG_PERIOD:]
                best_pair = [1e99, None, None, None, None, ] # err, i,j,k,a,b
                for i in range(self.AVG_PERIOD-3):
                    for j in range(i+3, self.AVG_PERIOD):
                        a = (calc_chars[j].y - calc_chars[i].y) / (calc_chars[j].x - calc_chars[i].x)
                        b = (calc_chars[i].y * calc_chars[j].x - calc_chars[i].x * calc_chars[j].y) / (calc_chars[j].x - calc_chars[i].x)
                        ks = [k for k in range(self.AVG_PERIOD) if k!=i and k!= j]
                        errors = [abs(calc_chars[k].x * a + b - calc_chars[k].y) for k in ks]
                        min_err = min(errors)
                        if min_err < best_pair[0]:
                            idx_if_min = min(range(len(errors)), key=errors.__getitem__)
                            best_pair = min_err, i, j, ks[idx_if_min], a, b
                _, i, j, k, a, b = best_pair
                steps_between = j - i + sum([ch.spaces_before for ch in calc_chars[i+1: j+1]])
                step = (calc_chars[j].x-calc_chars[i].x)/steps_between
                reference_x = (calc_chars[j].x + calc_chars[i].x + steps_between*step)/2
                steps_to_reference = round((new_char.x - reference_x) / step)
                expected_x = reference_x + step*steps_to_reference
                expected_y = expected_x * a + b
                w, h = (calc_chars[i].w + calc_chars[j].w + calc_chars[k].w)/3, (calc_chars[i].h + calc_chars[j].h + calc_chars[k].h)/3
                new_char.refined_box = [expected_x-w/2, expected_y-h/2, expected_x+w/2, expected_y+h/2]
                new_char.spaces_before = max(0, round( (expected_x - 0.5*(self.chars[-1].refined_box[0] + self.chars[-1].refined_box[2]))/step )-1)
                self.x, self.y = expected_x, expected_y
                self.h = h
                self.slip = a
            else:
                assert len(self.chars) >= 1
                step = new_char.h * self.SPACE_TO_H
                new_char.spaces_before = max(0, round((new_char.x - self.chars[-1].x)/step)-1)
                self.x = new_char.x
                self.y = new_char.y
                self.h = new_char.h
                self.slip = 0
            self.chars.append(new_char)
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
