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
logger = logging.getLogger('cfl.preprocessing')


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


def remove_stops(iterator, stopwords=set(), punctuation=True, digits=True,
                 whitespace=True):
    if not isinstance(stopwords, set):
        stopwords = set(stopwords)

    if punctuation:
        stopwords.update(string.punctuation)

    if digits:
        stopwords.update(string.digits)

    if whitespace:
        stopwords.update(string.whitespace)

    stopwords.update([''])
    for word in filter(lambda x: x not in stopwords, iterator):
        try:
            int(word)
            float(word)
        except ValueError:
            yield word

FOX_STOPS = set(
    """ a about above across after again against all almost alone along already
    also although always among an and another any anybody anyone anything
    anywhere are area areas around as ask asked asking asks at away b back
    backed backing backs be because become becomes became been before began
    behind being beings best better between big both but by c came can cannot
    case cases certain certainly clear clearly come could d did differ different
    differently do does done down downed downing downs during e each early
    either end ended ending ends enough even evenly ever every everybody
    everyone everything everywhere f face faces fact facts far felt few find
    finds first for four from full fully further furthered furthering furthers
    g gave general generally get gets give given gives go going good goods got
    great greater greatest group grouped grouping groups h had has have having
    he her herself here high higher highest him himself his how however i if
    important in interest interested interesting interests into is it its itself
    j just k keep keeps kind knew know known knows l large largely last later
    latest least less let lets like likely long longer longest m made make
    making man many may me member members men might more most mostly mr mrs much
    must my myself n necessary need needed needing needs never new newer newest
    next no non not nobody noone nothing now nowhere number numbered numbering
    numbers o of off often old older oldest on once one only open opened opening
    opens or order ordered ordering orders other others our out over p part
    parted parting parts per perhaps place places point pointed pointing points
    possible present presented presenting presents problem problems put puts
    q quite r rather really right room rooms s said same saw say says second
    seconds see sees seem seemed seeming seems several shall she should show
    showed showing shows side sides since small smaller smallest so some
    somebody someone something somewhere state states still such sure t take
    taken than that the their them then there therefore these they thing things
    think thinks this those though thought thoughts three through thus to today
    together too took toward turn turned turning turns two u under until up upon
    us use uses used v very w want wanted wanting wants was way ways we well
    wells went were what when where whether which while who whole whose why will
    with within without work worked working works would x y year years yet you
    young younger youngest your yours z """.split())

JAVA_RESERVED = set(
    """ abstract assert boolean break byte case catch char class const continue
    default do double else enum extends false final finally float for goto if
    implements import instanceof int interface long native new null package
    private protected public return short static strictfp super switch
    synchronized this throw throws transient true try void volatile while """.split())
