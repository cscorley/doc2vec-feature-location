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
from gensim.models import LdaModel

import utils
from corpora import ChangesetCorpus, TaserSnapshotCorpus, CorpusCombiner

def cli():
    logger.info("test")

    name = sys.argv[1]
    verbose = False
    project = None

    logging.basicConfig(format='%(asctime)s : %(levelname)s : ' +
                        '%(name)s : %(funcName)s : %(message)s')

    if verbose:
        logging.root.setLevel(level=logging.DEBUG)
    else:
        logging.root.setLevel(level=logging.INFO)

    with open("projects.csv", 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        customs = ['data_path']
        Project = namedtuple('Project',  ' '.join(header + customs))
        # figure out which column index contains the project name
        name_idx = header.index("name")
        version_idx = header.index("version")

        # find the project in the csv, adding it's info to config
        for row in reader:
            if name == row[name_idx]:
                # 🎶  do you believe in magicccccc
                # in a young girl's heart? 🎶
                row += (os.path.join('data', row[name_idx], row[version_idx], ''),)
                project = Project(*row)
                break

        # we can access project info by:
        #    project.name => "Blah Name"
        #    project.version => "0.22"

        if project is None:
            error("Could not find '%s' in 'projects.csv'!" % name)

    # reading in repos
    with open(os.path.join('data', project.name, 'repos.txt')) as f:
        repos = [line.strip() for line in f]

    print(project)
    repos_base = 'gits'
    if not os.path.exists(repos_base):
        utils.mkdir(repos_base)


    all_changes = CorpusCombiner()
    all_taser = CorpusCombiner()

    for url in repos:
        repo_name = url.split('/')[-1]
        target = os.path.join(repos_base, repo_name)
        print(target)
        try:
            repo = clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        changes = create_corpus(project, repo_name, repo, ChangesetCorpus)
        taser = create_corpus(project, repo_name, repo, TaserSnapshotCorpus)

        if changes:
            all_changes.add(changes)

        if taser:
            all_taser.add(taser)

    #write_out(project, all_taser)
    changeset_model = create_model(project, all_changes, 'Changeset')
    taser_model = create_model(project, all_taser, 'Taser')
    queries = load_queries(project)

    # to preprocess the queries use the corpus preprocessor!
    changes_queries = [ (q.id,
                         all_changes.id2word.doc2bow(
                             list(all_changes.preprocess(q.short)) +
                             list(all_changes.preprocess(q.long))))
                       for q in queries]

    taser_queries = [ (q.id,
                       all_taser.id2word.doc2bow(
                           list(all_taser.preprocess(q.short)) +
                           list(all_taser.preprocess(q.long))))
                     for q in queries]



def load_queries(project):
    with open(project.data_path + 'ids.txt') as f:
        ids = list(map(int, f.readlines()))

    queries = list()
    Query = namedtuple('Query', 'id short long')
    for id in ids:
        with open(project.data_path + 'Queries/ShortDescription' + str(id) + '.txt') as f:
            short = f.read()

        with open(project.data_path + 'Queries/LongDescription' + str(id) + '.txt') as f:
            long = f.read()

        queries.append(Query(id, short, long))

    return queries



def create_model(project, corpus, name):
    model_fname = project.data_path + name + '.lda'

    if not os.path.exists(model_fname):
        model = LdaModel(corpus,
                         id2word=corpus.id2word,
                         alpha='auto',
                         passes=1,
                         num_topics=100)

        model.save(model_fname)
    else:
        model = LdaModel.load(model_fname)

    return model


def write_out(project, all_taser):
    all_taser.metadata = True
    taserset = set(doc[1][0] for doc in all_taser)
    all_taser.metadata = False


    goldset_fname = project.data_path + 'allgold.txt'
    goldset = set()
    with open(goldset_fname) as f:
        for line in f:
            goldset.add(line.strip())

    print(len(goldset), len(taserset), len(taserset & goldset))

    missing_fname = project.data_path + 'missing-gold.txt'
    with open(missing_fname, 'w') as f:
        for each in sorted(list(goldset - taserset)):
            f.write(each + '\n')

    ours_fname = project.data_path + 'allours.txt'
    with open(ours_fname, 'w') as f:
        for each in sorted(list(taserset)):
            f.write(each + '\n')



def create_corpus(project, repo_name, repo, Kind):
    corpus_fname = project.data_path + Kind.__name__ + repo_name + '.mallet'
    if not os.path.exists(corpus_fname):
        try:
            if project.sha:
                corpus = Kind(repo, project.sha, lazy_dict=True)
            else:
                corpus = Kind(repo, project.ref, lazy_dict=True)
        except KeyError:
            return # nothing to see here, move along
        except TaserError:
            return

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
