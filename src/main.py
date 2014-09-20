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

import scipy.stats

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
                # build the data_path value
                row += (os.path.join('data', row[name_idx], row[version_idx], ''),)

                # try to convert string values to numbers
                for idx, item in enumerate(row):
                    # try int first, floats will throw an error
                    try:
                        item = int(item)
                        row[idx] = item
                        continue
                    except ValueError:
                        pass

                    # try float second
                    try:
                        item = float(item)
                        row[idx] = item
                    except ValueError:
                        pass

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

    # combine dictionaries!
    all_changes.id2word.merge_with(all_taser.id2word)
    all_taser.id2word = all_changes.id2word

    #write_out_missing(project, all_taser)
    changeset_model = create_model(project, all_changes, 'Changeset')
    taser_model = create_model(project, all_taser, 'Taser')

    # create/load document lists
    queries = load_queries(project)
    goldsets = load_goldsets(project)

    # to preprocess the queries use the corpus preprocessor!
    query_bows = [ (all_changes.id2word.doc2bow(
                    list(all_changes.preprocess(q.short)) +
                    list(all_changes.preprocess(q.long))),
                    q.id)
                for q in queries]

    # get the query topic
    changeset_query_topic = get_topics(changeset_model, query_bows)
    taser_query_topic = get_topics(taser_model, query_bows)

    # get the doc topic for the methods of interest
    changeset_doc_topic = get_topics(changeset_model, all_taser)
    taser_doc_topic = get_topics(taser_model, all_taser)

    # get the ranks
    changeset_ranks = get_rank(changeset_query_topic, changeset_doc_topic)
    taser_ranks = get_rank(taser_query_topic, taser_doc_topic)

    # calculate MRR
    changeset_first_rels = get_frms(goldsets, changeset_ranks)
    taser_first_rels = get_frms(goldsets, taser_ranks)
    print(len(goldsets), len(changeset_first_rels))
    print(len(taser_first_rels))

    changeset_mrr = (1.0/float(len(changeset_first_rels)) *
                    sum(1.0/float(num) for num, q, d in changeset_first_rels))
    taser_mrr = (1.0/float(len(taser_first_rels)) *
                    sum(1.0/float(num) for num, q, d in taser_first_rels))

    print('changeset mrr:', changeset_mrr)
    print('taser mrr:', taser_mrr)

    with open(project.data_path + 'changeset_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(changeset_first_rels)

    with open(project.data_path + 'taser_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(taser_first_rels)

    first_rels = dict()

    for num, query_id, doc_meta in changeset_first_rels:
        if query_id not in first_rels:
            first_rels[query_id] = [num]
        else:
            print('duplicate qid found:', query_id)

    for num, query_id, doc_meta in taser_first_rels:
        if query_id not in first_rels:
            print('qid not found:', query_id)
        else:
            first_rels[query_id].append(num)

    x = [v[0] for v in first_rels.values()]
    y = [v[1] for v in first_rels.values()]

    print('ranksums:', scipy.stats.ranksums(x, y))
    print('mann-whitney:', scipy.stats.mannwhitneyu(x, y))
    print('wilcoxon signedrank:', scipy.stats.wilcoxon(x, y))
    print('friedman:', scipy.stats.friedmanchisquare(*first_rels.values()))



def get_frms(goldsets, ranks):
    frms = list()
    for gid, gset in goldsets:
        for query_id, rank in ranks:
            if gid == query_id:
                for idx, metadist in enumerate(rank):
                    doc_meta, dist = metadist
                    d_name, d_repo = doc_meta
                    if d_name in gset:
                        frms.append((idx+1, query_id, doc_meta))
                        break
                break
    return frms

def get_rank(query_topic, doc_topic, distance_measure=utils.hellinger_distance):
    ranks = list()
    for q_meta, query in query_topic:
        q_dist = list()

        for d_meta, doc in doc_topic:
            distance = distance_measure(query, doc)
            q_dist.append((d_meta, 1.0 - distance))

        ranks.append((q_meta, sorted(q_dist, reverse=True, key=lambda x: x[1])))

    return ranks


def get_topics(model, corpus):
    doc_topic = list()
    if hasattr(corpus, 'metadata'):
        corpus.metadata = True
    for doc, metadata in corpus:
        topics = model.__getitem__(doc, eps=0)
        topics = [val for id, val in topics]
        doc_topic.append((metadata, list(sorted(topics, reverse=True))))

    return doc_topic

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

def load_goldsets(project):
    with open(project.data_path + 'ids.txt') as f:
        ids = list(map(int, f.readlines()))

    goldsets = list()
    Gold = namedtuple('Gold', 'id golds')
    for id in ids:
        with open(project.data_path + 'GoldSets/GoldSet' + str(id) + '.txt') as f:
            golds = set(x.strip() for x in f.readlines())

        goldsets.append(Gold(id, golds))

    return goldsets



def create_model(project, corpus, name):
    model_fname = project.data_path + name + '.lda'

    if not os.path.exists(model_fname):
        model = LdaModel(corpus,
                         id2word=corpus.id2word,
                         alpha=project.alpha,
                         passes=project.passes,
                         num_topics=project.num_topics)

        model.save(model_fname)
    else:
        model = LdaModel.load(model_fname)

    return model


def write_out_missing(project, all_taser):
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
