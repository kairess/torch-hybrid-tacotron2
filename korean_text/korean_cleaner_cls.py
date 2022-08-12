from korean_text.korean_utils import text_normalize, text_to_id, id_to_text, unit_normalize
from korean_text.dicManager import DicManager
import jamo

class KoreanCleaner:
    def __init__(self, sentence_dictionary=None, term_dictionary=None):
        self._sentence_dictionary = sentence_dictionary
        self._term_dictionary = term_dictionary
        self._dict_manager = DicManager(self.sentence_dictionary, self.term_dictionary)
    # end def

    @property
    def sentence_dictionary(self):
        return self._sentence_dictionary

    @property
    def term_dictionary(self):
        return self._term_dictionary

    @sentence_dictionary.setter
    def sentence_dictionary(self, sentence_dict):
        self._sentence_dictionary = sentence_dict

    # end def

    @term_dictionary.setter
    def term_dictionary(self, term_dict):
        self._term_dictionary = term_dict

    # end def

    def apply(self):
        self._dict_manager = DicManager(self._sentence_dictionary, self._term_dictionary)
    # end def

    def clean_text(self, text):
        text = text_normalize(text, self._dict_manager)
        sequence = text_to_id(text)
        text = id_to_text(sequence, combine=True)
        return text
    # end def
# end class


if __name__ == '__main__':
    cleaner = KoreanCleaner(None, None)
    text = '0.000001'
    result = cleaner.clean_text(text)
    print(result)
    # for i in range(99):
    #     text = str(i+1) + '명'
    #     print(cleaner.clean_text(text)[:-1] + '\t' + str(i+1))

