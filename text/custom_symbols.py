""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_eos = '~'
_pad = '_'
# _punctuation = '!\'(),.:;? '
_punctuation = '!\',.?'
_space = ' '
_special = '-'
_letters = 'abcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# korean
_JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
_JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
_JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

_VALID_JAMO = [jamo for jamo in _JAMO_LEADS + _JAMO_VOWELS + _JAMO_TAILS]


# Export all symbols:
symbols = [_eos] + [_pad] + list(_special) + list(_punctuation) + [_space] + list(_letters) + _arpabet + _VALID_JAMO
