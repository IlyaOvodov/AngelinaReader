from pathlib import Path
from collections import defaultdict, OrderedDict

if __name__=='__main__':
    root_path = r'D:\_Backups\Книги'
    target_path = r'D:\Programming\Braille\MyCode\pseudo_labeling'


def process_line(ln, mono_cnt, bi_cnt):
    ln = ln.strip()
    for i, ch in enumerate(ln):
        mono_cnt[ch] += 1
        if i == 0:
            bi_cnt[' ' + ch] += 1
        else:
            bi_cnt[ln[i-1] + ch] += 1
    if ln:
        bi_cnt[ch + ' '] += 1


def process_file(fn, mono_cnt, bi_cnt):
    print(fn)
    lines = None
    try:
        with open(fn, encoding='utf-8') as f:
            lines = f.readlines()
    except:
        with open(fn, encoding='cp1251') as f:
            lines = f.readlines()
    x=1
    for ln in lines:
        process_line(ln, mono_cnt, bi_cnt)


def norm(d):
    s = sum(d.values())
    d = OrderedDict({k:v/s for k,v, in sorted([(k,v) for k,v in d.items()], key=lambda x:-x[1])})
    return d


def process(root_path):
    files = Path(root_path).glob('**/*.txt')
    mono_cnt = defaultdict(int)
    bi_cnt = defaultdict(int)
    for i, fn in enumerate(files):
        process_file(fn, mono_cnt, bi_cnt)
        # if i==4:
        #     break
    print('total_files:', i+1)
    mono_cnt = norm(mono_cnt)
    bi_cnt = norm(bi_cnt)
    return mono_cnt, bi_cnt

def save(fn, d):
    with open(fn, 'w', encoding='utf-8-sig') as f:
        for k, v in d.items():
            f.write('{} {}\n'.format(k,v))


def test():
    def test_ln(ln):
        mono_cnt = defaultdict(int)
        bi_cnt = defaultdict(int)
        process_line(ln, mono_cnt, bi_cnt)
        print(mono_cnt)
        print(bi_cnt)

    test_ln('qwe')
    test_ln('')
    test_ln('qqweeqw')


if __name__=='__main__':
    #test()
    m, b = process(root_path)
    save(Path(target_path)/'mono_freq.txt', m)
    save(Path(target_path)/'bi_freq.txt', b)

    print(sum(b.values()),sum(m.values()))
    print(len(m), m)
    print(len(b),  b)


