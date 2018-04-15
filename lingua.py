# -*- coding: utf-8 -*-

import re

from gensim.models import KeyedVectors
from nltk.tokenize import TreebankWordTokenizer
from pymorphy2 import MorphAnalyzer

DEFAULT_ENCODING = "UTF-8"
RE_RUS_LETTERS = re.compile(u"[^абвгдеёжзийклмнопрстуфхцчшщъыьэюя ]")

TRANSIT = {
    "ADJF": "ADJ",
    "ADJS": "ADJ",
    "ADVB": "ADV",
    "COMP": "ADV",
    "CONJ": "CCONJ",
    "GRND": "VERB",
    "INFN": "VERB",
    "INTJ": "INTJ",
    "LATN": "X",
    "NOUN": "NOUN",
    "NPRO": "PRON",
    "NUMB": "NUM",
    "NUMR": "NUM",
    "PNCT": "PUNCT",
    "PRCL": "PART",
    "PRED": "ADV",
    "PREP": "ADP",
    "PRTF": "ADJ",
    "PRTS": "VERB",
    "ROMN": "X",
    "SYMB": "SYM",
    "UNKN": "X",
    "VERB": "VERB"
}
RE_TRANSIT = re.compile("|".join(TRANSIT.keys()))

MODEL_PATH = "data/ruwikiruscorpora_0_300_20.bin"


class Text(object):
    """???

    """
    __tokenizer = TreebankWordTokenizer()

    def __init__(self, s):
        self.__value = s.decode(DEFAULT_ENCODING)

    def __str__(self):
        return self.__value.encode(DEFAULT_ENCODING)

    def tokenize(self):
        """Tokenizes text.

        Returns:
            List of word.

        """
        text = RE_RUS_LETTERS.sub(u"", self.__value.lower())

        words = []
        for word in self.__tokenizer.tokenize(text):
            words.append(Word(word))

        return words


class Word(object):
    """???

    """
    __morph = MorphAnalyzer()
    __model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

    def __init__(self, s):
        self.__value = s

    def __str__(self):
        return self.__value.encode(DEFAULT_ENCODING)

    @property
    def is_meaningful(self):
        return True

    @property
    def normal_form(self):
        return self.__morph.parse(self.__value)[0].normal_form

    @property
    def part_of_speech(self):
        return RE_TRANSIT.sub(lambda m: TRANSIT[m.group(0)], str(self.__morph.parse(self.__value)[0].tag.POS))

    def lemmatize(self):
        return u"{}_{}".format(self.normal_form, self.part_of_speech)

    def vectorize(self):
        """Vectorizes word.

        Returns:
            Vectorized word.

        """
        word = self.lemmatize()
        if word in self.__model:
            return self.__model[word]
