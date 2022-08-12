from jamo import hangul_to_jamo
from korean_text.symbols import ALL_SYMBOLS, SPECIAL_CHARS, UNK
import korean_text.korean as korean

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}


def text_normalize(text, dictionary):
    text = korean.format(text, dictionary)
    text = korean.clean(text)
    text = korean.normalize(text, dictionary)
    return text
#end def

def unit_normalize(text, dictionary):
    text = korean.normalize_unit(text, dictionary)
    return text
# end def

def text_to_id(text):
    tokens = list(hangul_to_jamo(text))
    return [char_to_id[token] if token in char_to_id else char_to_id[UNK] for token in tokens if not _is_special_char(token)]
#end def

def id_to_text(sequece, combine=False):
    if combine:
        return "".join([id_to_char[token] for token in sequece])
    else:
        return [id_to_char[token] for token in sequece]
#end def

def _is_special_char(s):
    return s in SPECIAL_CHARS
#end def


if __name__ == '__main__':
    text = text_normalize("나그네는 나ㅁ산")
    tti = text_to_id(text)
    print(tti)
    itt = id_to_text(tti, combine=True)
    print(itt)