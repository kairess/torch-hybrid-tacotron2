
JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

PUNC = ",.?!"
SPACE = ' '

SOS = "@"
PAD = '_'
EOS = '~'
UNK = "#"

VALID_JAMO = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS
VALID_CHARS = VALID_JAMO + PUNC + SPACE
SPECIAL_CHARS = SOS + EOS + PAD + UNK

ALL_SYMBOLS = SPECIAL_CHARS + VALID_CHARS
