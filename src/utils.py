#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

import math
import logging
import os
import sys


logger = logging.getLogger('cfl.utils')


def kullback_leibler_divergence(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    divergence = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        if q > 0.0 and p > 0.0:
            divergence += q * math.log10(q / p)

    return divergence


def hellinger_distance(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    distance = 0.0
    z = zip(q_dist, p_dist)
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        inner = math.sqrt(q) - math.sqrt(p)
        distance += (inner * inner)

    distance /= 2
    distance = math.sqrt(distance)
    return distance


def cosine_distance(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    numerator = 0.0
    denominator_a = 0.0
    denominator_b = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        numerator += (q * p)
        denominator_a += (q * q)
        denominator_b += (p * p)

    denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
    similarity = (numerator / denominator)
    return 1.0 - similarity


def jensen_shannon_divergence(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    q_dist, p_dist, M = list(), list(), list()
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        M.append((q + p) / 2)
        q_dist.append(q)
        p_dist.append(p)

    divergence_a = (kullback_leibler_divergence(q_dist, M) / 2)
    divergence_b = (kullback_leibler_divergence(p_dist, M) / 2)
    return divergence_a + divergence_b


def total_variation_distance(q_dist, p_dist, filter_by=0.001):
    z = zip(q_dist, p_dist)
    distance = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        distance += math.fabs(q - p)

    distance /= 2
    return distance


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

# exception handling mkdir -p


def mkdir(d):
    try:
        os.makedirs(d)
    except os.error as e:
        if 17 == e.errno:
            # the directory already exists
            pass
        else:
            print('Failed to create "%s" directory!' % d)
            sys.exit(e.errno)

