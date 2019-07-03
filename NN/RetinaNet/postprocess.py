import DSBI_invest.data
import DSBI_invest.letters


class LineChar:
    def __init__(self, box, label, spaces_before):
        self.box = box
        self.label = label
        self.spaces_before = spaces_before
        self.char = ''


class Line:
    SPACES_THR = 0.4
    LINE_THR = 0.5
    def __init__(self, box, label):
        self.chars = [LineChar(box, label, 0)]
        self.y = (box[1] + box[3])/2
        self.h = box[3]-box[1]
        #self.spaces = []
    def check_and_append(self, box, label):
        y = (box[1] + box[3])/2
        if abs(self.y-y)/self.h < self.LINE_THR:
            r = self.chars[-1].box[2]
            #spaces = (box[0]-r)/self.h
            #self.spaces.append(spaces)
            spaces = int((box[0]-r)/self.h + self.SPACES_THR)
            self.chars += [LineChar(box, label, spaces)]
            h = box[3] - box[1]
            n = min(len(self.chars),5)
            self.y += (y - self.y)/n
            self.h += (h - self.h)/n
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
        for ch in l.chars:
            if ch.spaces_before:
                digit_mode = False
            lbl = DSBI_invest.data.int_to_letter(ch.label.item(), 'NUM' if digit_mode else lang)
            if lbl == DSBI_invest.letters.num_sign:
                digit_mode = True
                ch.char = ''
            else:
                if not lbl:
                    digit_mode = False
                    lbl = DSBI_invest.data.int_to_label123(ch.label.item())
                    lbl = '&'+lbl
                ch.char = lbl
    return lines



if __name__ == '__main__':
    norm_boxes([(1,2,3,4)], [1])
