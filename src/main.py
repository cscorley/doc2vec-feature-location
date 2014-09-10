#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

from __future__ import print_function

import logging
logger = logging.getLogger('cfl.main')

import sys
import os.path
import csv
from collections import namedtuple

import dulwich
import dulwich.client
import dulwich.repo

from gensim.corpora import MalletCorpus, Dictionary

import utils
from corpora import ChangesetCorpus, TaserSnapshotCorpus

def cli():
    print("test")
    logger.info("test")

    name = sys.argv[1]

    project = None

    with open("projects.csv", 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        Project = namedtuple('Project',  ' '.join(header))

        # figure out which column index contains the project name
        name_idx = header.index("name")

        # find the project in the csv, adding it's info to config
        for row in reader:
            if name == row[name_idx]:
                # 🎶  do you believe in magicccccc
                # in a young girl's heart? 🎶
                project = Project(*row)
                break

        # we can access project info by:
        #    project.name => "Blah Name"
        #    project.version => "0.22"

        if project is None:
            error("Could not find '%s' in 'projects.csv'!" % name)

    # reading in repos
    repos = list()
    with open(os.path.join('data', project.name, 'repos.txt')) as f:
        for line in f:
            repos.append(line.strip())

    print(project)
    print(repos)
    repos_base = 'gits'
    if not os.path.exists(repos_base):
        utils.mkdir(repos_base)

    ourset = set()

    for url in repos:
        repo_name = url.split('/')[-1]
        target = os.path.join(repos_base, repo_name)
        print(target)
        try:
            repo = clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        changes = create_corpus(project, repo_name, repo, ChangesetCorpus)
        try:
            taser = create_corpus(project, repo_name, repo, TaserSnapshotCorpus)
        except Exception:
            pass

        taser.metadata = True
        ourset.update(set(doc[1][0] for doc in c))
        taser.metadata = False

    goldset_fname = os.path.join('data', project.name, 'v' + project.version, 'allgold.txt')
    goldset = set()
    with open(goldset_fname) as f:
        for line in f:
            goldset.add(line.strip())

    print(len(goldset), len(ourset), len(ourset & goldset))
    missing_fname = os.path.join('data', project.name, 'v' + project.version, 'missing-gold.txt')
    with open(missing_fname, 'w') as f:
        for each in sorted(list(goldset - ourset)):
            f.write(each + '\n')



def create_corpus(project, repo_name, repo, Kind):
    corpus_fname = os.path.join('data', project.name, Kind.__name__ + repo_name + '.mallet')
    if not os.path.exists(corpus_fname):
        try:
            if project.sha:
                corpus = Kind(repo, project.sha, lazy_dict=True)
            else:
                corpus = Kind(repo, project.ref, lazy_dict=True)
        except KeyError:
            return # nothing to see here, move along

        corpus.metadata = True
        MalletCorpus.serialize(corpus_fname, corpus,
                               id2word=corpus.id2word, metadata=True)
        corpus.metadata = False
        corpus.id2word.save(corpus_fname + '.dict')
    else:
        if os.path.exists(corpus_fname + '.dict'):
            id2word = Dictionary.load(corpus_fname + '.dict')
        else:
            id2word = None

        corpus = MalletCorpus(corpus_fname, id2word=id2word)

    return corpus


def clone(source, target, bare=False):
    client, host_path = dulwich.client.get_transport_and_path(source)

    if target is None:
        target = host_path.split("/")[-1]

    if not os.path.exists(target):
        os.mkdir(target)

    if bare:
        r = dulwich.repo.Repo.init_bare(target)
    else:
        r = dulwich.repo.Repo.init(target)

    remote_refs = client.fetch(host_path, r,
                               determine_wants=r.object_store.determine_wants_all)

    r["HEAD"] = remote_refs["HEAD"]

    for key, val in remote_refs.iteritems():
        if not key.endswith('^{}'):
            r.refs.add_if_new(key, val)

    return r
