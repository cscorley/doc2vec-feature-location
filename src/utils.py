#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

import logging
import os
import sys
import numpy
import scipy
import scipy.spatial


logger = logging.getLogger('cfl.utils')

SQRT2 = numpy.sqrt(2)


def calculate_mrr(p):
    vals = list()
    for item in p:
        if item:
            vals.append(1.0/item)
        else:
            vals.append(0.0)

    return numpy.mean(vals)


def hellinger_distance(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return scipy.linalg.norm(numpy.sqrt(p) - numpy.sqrt(q)) / SQRT2


def kullback_leibler_divergence(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return scipy.stats.entropy(p, q)


def cosine_distance(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return scipy.spatial.distance.cosine(p, q)


def jensen_shannon_divergence(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    M = (p + q)/2
    return (kullback_leibler_divergence(p, M) +
            kullback_leibler_divergence(p, M)) / 2


def total_variation_distance(p, q):
    p = numpy.array(p)
    q = numpy.array(q)
    return numpy.sum(numpy.abs(p - q)) / 2


def score(model, fn):
    # thomas et al 2011 msr
    #
    scores = list()
    for a, topic_a in norm_phi(model):
        score = 0.0
        for b, topic_b in norm_phi(model):
            if a == b:
                continue

            score += fn(topic_a, topic_b)

        score *= (1.0 / (model.num_topics - 1))
        logger.debug("topic %d score %f" % (a, score))
        scores.append((a, score))

    return scores


def norm_phi(model):
    for topicid in range(model.num_topics):
        topic = model.state.get_lambda()[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        yield topicid, topic


def mkdir(d):
    # exception handling mkdir -p
    try:
        os.makedirs(d)
    except os.error as e:
        if 17 == e.errno:
            # the directory already exists
            pass
        else:
            print('Failed to create "%s" directory!' % d)
            sys.exit(e.errno)
