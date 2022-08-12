from jamo import hangul_to_jamo

PAD = '_'
EOS = '~'
PUNC = ',.?!'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

KOREAN_SYMBOLS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS
VALID_CHARS = KOREAN_SYMBOLS + PUNC + SPACE
ALL_SYMBOLS = PAD + EOS + VALID_CHARS

_symbol_to_id = {s: i for i, s in enumerate(ALL_SYMBOLS)}
_id_to_symbol = {i: s for i, s in enumerate(ALL_SYMBOLS)}


def text_to_sequence(text):
	clean_text = clean(text)
	sequence = symbols_to_sequence(tokenize(clean_text))
	return sequence

def symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols]

def sequence_to_symbols(sequence):
    return [_id_to_symbol[s] for s in sequence]

def tokenize(text):
    tokens = list(hangul_to_jamo(text))
    return [token for token in tokens] + [EOS]

def clean(text):
	return text.replace('#', '')


if __name__ == '__main__':

	text = "안녕하세요 AI Labs 입니다."
	a = "".join(hangul_to_jamo(text))
	# print(text_to_sequence(text))
	print(a)
	for c in a:
		print(c)


