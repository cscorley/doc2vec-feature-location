#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

"""
Code for splitting the terms.
"""

import string

import logging
logger = logging.getLogger('flt.preprocessing')


def tokenize(s):
    return s.split()


def to_unicode(document, info=[]):
    document = document.replace('\x00', ' ')  # remove nulls
    document = document.strip()
    if not isinstance(document, unicode):
        for codec in ['utf8', 'latin1', 'ascii']:
            try:
                return unicode(document, encoding=codec)
            except UnicodeDecodeError as e:
                logger.debug('%s %s %s' % (codec, str(e), ' '.join(info)))

    return document


def split(iterator):
    for token in iterator:
        word = u''
        for char in token:
            if char.isupper() and all(map(lambda x: x.isupper(), word)):
                # keep building if word is currently all uppercase
                word += char

            elif char.islower() and all(map(lambda x: x.isupper(), word)):
                # stop building if word is currently all uppercase,
                # but be sure to take the first letter back
                if len(word) > 1:
                    yield word[:-1]
                    word = word[-1]

                word += char

            elif char.islower() and any(map(lambda x: x.islower(), word)):
                # keep building if the word is has any lowercase
                # (word came from above case)
                word += char

            elif char.isdigit() and all(map(lambda x: x.isdigit(), word)):
                # keep building if all of the word is a digit so far
                word += char

            elif char in string.punctuation:
                if len(word) > 0:
                    yield word
                    word = u''

                # always yield punctuation as a single token
                yield char

            else:
                if len(word) > 0:
                    yield word

                word = char

        if len(word) > 0:
            yield word


def remove_stops(iterator, stopwords=set()):
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)

    stopwords.update(string.punctuation)
    stopwords.update(string.digits)
    stopwords.update(string.whitespace)
    stopwords.update([''])
    for word in filter(lambda x: x not in stopwords, iterator):
        try:
            int(word)
            float(word)
        except ValueError:
            yield word


def read_stops(l):
    stops = list()
    for each in l:
        with open(each) as f:
            stops.extend(f.readlines())

    return set([word.strip() for word in stops])
