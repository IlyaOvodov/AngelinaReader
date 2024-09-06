# if __name__=="__main__":
#     import sys
#     sys.path.append(".")
from pathlib import Path
import louis
import local_config
from braille_utils import label_tools as lt

liblouis_tables_path_prefix = getattr(local_config, 'liblouis_tables_path_prefix', '')
liblouis_tables = {
    "EN2": "en-ueb-g2.ctb",
}

def update_word_at_line(line, start_idx, end_idx, liblouis_word, existing_word):
    """
    Update word in line if liblouis generated word differes from existing_word
    """
    if liblouis_word != existing_word:
        for ch in line.chars[start_idx:end_idx]:
            ch.labeling_char = '~' + lt.int_to_label123(ch.label)
            ch.char = ""
        line.chars[start_idx].char = liblouis_word


def interpret_line_liblouis_word_by_word(line, tables, mode):
    """
    Update line using liblouis word by word
    Use this function if interpret_line_liblouis_as_a_whole result in text that can't be
    associated with words in line
    """
    braille_word = ""
    curr_word = ""
    init_word_ch_idx = 0
    for i, ch in enumerate(line.chars):
        if ch.spaces_before:
            if braille_word:
                liblouis_word = louis.backTranslateString(tables, braille_word)
                update_word_at_line(line, init_word_ch_idx, i, liblouis_word, curr_word)
                braille_word = ""
                curr_word = ""
                init_word_ch_idx = i
        braille_word += lt.int_to_unicode(ch.label)
        curr_word += ch.char or ""
    if braille_word:
        liblouis_word = louis.backTranslateString(tables, braille_word)
        update_word_at_line(line, init_word_ch_idx, len(line.chars), liblouis_word, curr_word)
    return mode

def interpret_line_liblouis_as_a_whole(line, tables, mode):
    """
    Update line using liblouis.
    Keeps existing interpretation for chars if liblouis doesn't change word
    :param line: postprocess.Line
    :param lang: "EN2" etc., one of supported languages enumerated in liblouis_tables dict
    :param mode: not used.
    :return: mode
    """
    whole_braille_string = ""
    word_starts = [0]
    existing_words = []
    curr_word = ""
    for i, ch in enumerate(line.chars):
        if ch.spaces_before and  i != 0:
            existing_words.append(curr_word)
            curr_word = ""
            whole_braille_string += " "
            word_starts.append(i)
        whole_braille_string += lt.int_to_unicode(ch.label)
        curr_word += ch.char or ""
    existing_words.append(curr_word)
    word_starts.append(len(line.chars))  # fictive end to limit last word
    translation = louis.backTranslateString(tables, whole_braille_string)
    words = translation.split(" ")
    if len(words) == len(word_starts) - 1:
        for i, word in enumerate(words):
            update_word_at_line(line, word_starts[i], word_starts[i+1], word, existing_words[i])
    else:
        return interpret_line_liblouis_word_by_word(line, tables, mode)
    return mode

def interpret_line_liblouis(line, lang, mode = None):
    """
    Interprets line using Liblouis library
    Fills line chars
    :param line: postprocess.Line
    :param lang: "EN2" etc., one of supported languages enumerated in liblouis_tables dict
    :param mode: not used.
    :return: mode
    """
    liblouis_table = liblouis_tables.get(lang)
    assert liblouis_table is not None, f"Translation table is not defined for language {lang}"
    liblouis_tables_list = [liblouis_tables_path_prefix + liblouis_table]
    return interpret_line_liblouis_as_a_whole(line, liblouis_tables_list, mode)

if __name__=="__main__":
    from data_utils import data as data
    import postprocess

    lang = 'EN2'
    json_filename = r"D:\Programming.Data\Braille\Книги на английском от Оксаны распознанные\v3\IMG_20210827_161627.labeled.json"

    # for performance test
    liblouis_table = liblouis_tables.get(lang)
    assert liblouis_table is not None, f"Translation table is not defined for language {lang}"
    liblouis_tables_list = [str(liblouis_tables_path_prefix + liblouis_table)]
    rects = data.read_LabelMe_annotation(label_filename = json_filename, get_points = False)
    boxes = [r[:4] for r in rects]
    labels = [r[4] for r in rects]
    lines = postprocess.boxes_to_lines(boxes, labels, lang='EN')
    for i in range(1000):
        for ln in lines:
            interpret_line_liblouis(ln, lang, None)
            interpret_line_liblouis_word_by_word(ln, liblouis_tables_list, None)
            postprocess.interpret_line_RU(ln, lang, None)
    print("Done")

