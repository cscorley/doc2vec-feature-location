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
from corpora import (ChangesetCorpus, TaserSnapshotCorpus, TaserReleaseCorpus,
                     CorpusCombiner, GeneralCorpus)
from errors import TaserError

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
        customs = ['data_path', 'src_path']
        Project = namedtuple('Project',  ' '.join(header + customs))
        # figure out which column index contains the project name
        name_idx = header.index("name")
        version_idx = header.index("version")

        # find the project in the csv, adding it's info to config
        for row in reader:
            if name == row[name_idx]:

                # build the data_path value
                row += (os.path.join('data', row[name_idx], row[version_idx], ''),)

                # build the src_path value
                row += (os.path.join('data', row[name_idx], row[version_idx], 'src'),)

                # try to convert string values to numbers
                for idx, item in enumerate(row):
                    if item:
                        # try int first, floats will throw an error here
                        try:
                            row[idx] = int(item)
                            continue
                        except ValueError:
                            pass

                        # try float second
                        try:
                            row[idx] = float(item)
                        except ValueError:
                            pass
                    else:
                        row[idx] = None

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
        repo_urls = [line.strip() for line in f]

    print(project)
    repos_base = 'gits'
    if not os.path.exists(repos_base):
        utils.mkdir(repos_base)


    repos = list()

    for url in repo_urls:
        repo_name = url.split('/')[-1]
        target = os.path.join(repos_base, repo_name)
        print(target)
        try:
            repo = clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        repos.append(repo)

    # get corpora
    id2word = create_dictionary(project)

    logger.info('PRIMARY DICTIONARY %s', id(id2word))

    changeset_corpus = create_corpus(project, repos, id2word, ChangesetCorpus)
    snapshot_corpus = create_corpus(project, repos, id2word, TaserSnapshotCorpus)
    release_corpus = create_release_corpus(project, id2word)

    assert id2word is changeset_corpus.id2word
    assert id2word is snapshot_corpus.id2word
    assert id2word is release_corpus.id2word

    # create/load document lists
    queries = create_queries(project, id2word)
    goldsets = load_goldsets(project)

    #write_out_missing(project, snapshot_corpus)
    changeset_model = create_model(project, changeset_corpus, id2word, 'Changeset')
    snapshot_model = create_model(project, snapshot_corpus, id2word,'Snapshot')
    release_model = create_model(project, release_corpus, id2word,'Release')

    # get the query topic
    changeset_query_topic = get_topics(changeset_model, queries)
    snapshot_query_topic = get_topics(snapshot_model, queries)
    release_query_topic = get_topics(release_model, queries)

    # get the doc topic for the methods of interest (git snapshot)
    changeset_doc_topic = get_topics(changeset_model, snapshot_corpus)
    snapshot_doc_topic = get_topics(snapshot_model, snapshot_corpus)

    # get the doc topic for the methods of interest (release)
    changeset2_doc_topic = get_topics(changeset_model, release_corpus)
    release_doc_topic = get_topics(release_model, release_corpus)

    # get the ranks
    changeset_ranks = get_rank(changeset_query_topic, changeset_doc_topic)
    snapshot_ranks = get_rank(snapshot_query_topic, snapshot_doc_topic)

    changeset2_ranks = get_rank(changeset_query_topic, changeset2_doc_topic)
    release_ranks = get_rank(release_query_topic, release_doc_topic)

    # calculate MRR
    changeset_first_rels = get_frms(goldsets, changeset_ranks)
    snapshot_first_rels = get_frms(goldsets, snapshot_ranks)

    changeset2_first_rels = get_frms(goldsets, changeset2_ranks)
    release_first_rels = get_frms(goldsets, release_ranks)

    changeset_mrr = (1.0/float(len(changeset_first_rels)) *
                    sum(1.0/float(num) for num, q, d in changeset_first_rels))
    snapshot_mrr = (1.0/float(len(snapshot_first_rels)) *
                    sum(1.0/float(num) for num, q, d in snapshot_first_rels))

    changeset2_mrr = (1.0/float(len(changeset2_first_rels)) *
                    sum(1.0/float(num) for num, q, d in changeset2_first_rels))
    release_mrr = (1.0/float(len(release_first_rels)) *
                    sum(1.0/float(num) for num, q, d in release_first_rels))

    print('changeset mrr:', changeset_mrr)
    print('snapshot mrr:', snapshot_mrr)

    print('changeset2 mrr:', changeset2_mrr)
    print('release mrr:', release_mrr)

    with open(project.data_path + 'changeset_' + str(project.num_topics) + '_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(changeset_first_rels)

    with open(project.data_path + 'snapshot_' + str(project.num_topics) + '_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(snapshot_first_rels)

    with open(project.data_path + 'changeset2_' + str(project.num_topics) + '_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(changeset2_first_rels)

    with open(project.data_path + 'release_' + str(project.num_topics) + '_frms.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(release_first_rels)

    first_rels = dict()

    for num, query_id, doc_meta in changeset_first_rels:
        if query_id not in first_rels:
            first_rels[query_id] = [num]
        else:
            logger.info('duplicate qid found:', query_id)

    for num, query_id, doc_meta in snapshot_first_rels:
        if query_id not in first_rels:
            logger.info('qid not found:', query_id)
        else:
            first_rels[query_id].append(num)

    for num, query_id, doc_meta in changeset2_first_rels:
        if query_id not in first_rels:
            logger.info('qid not found:', query_id)
        else:
            first_rels[query_id].append(num)

    for num, query_id, doc_meta in release_first_rels:
        if query_id not in first_rels:
            logger.info('qid not found:', query_id)
        else:
            first_rels[query_id].append(num)

    x = [v[0] for v in first_rels.values()]
    y = [v[1] for v in first_rels.values()]
    x2 = [v[2] for v in first_rels.values()]
    y2 = [v[3] for v in first_rels.values()]

    print('ranksums:', scipy.stats.ranksums(x, y))
    print('ranksums2:', scipy.stats.ranksums(x2, y2))

    print('mann-whitney:', scipy.stats.mannwhitneyu(x, y))
    print('mann-whitney2:', scipy.stats.mannwhitneyu(x2, y2))

    print('wilcoxon signedrank:', scipy.stats.wilcoxon(x, y))
    print('wilcoxon signedrank2:', scipy.stats.wilcoxon(x2, y2))

    print('friedman:', scipy.stats.friedmanchisquare(x, y, x2, y2))

def get_frms(goldsets, ranks):
    logger.info('Getting FRMS for %d goldsets and %d ranks',
                len(goldsets), len(ranks))
    frms = list()
    for g_id, goldset in goldsets:
        for q_meta, rank in ranks:
            query_id, _ = q_meta
            if g_id == query_id:
                for idx, metadist in enumerate(rank):
                    doc_meta, dist = metadist
                    d_name, d_repo = doc_meta
                    if d_name in goldset:
                        frms.append((idx+1, query_id, doc_meta))
                        break
                break

    logger.info('Returning %d FRMS', len(frms))
    return frms

def get_rank(query_topic, doc_topic, distance_measure=utils.hellinger_distance):
    logger.info('Getting ranks between %d query topics and %d doc topics',
                len(query_topic), len(doc_topic))
    ranks = list()
    for q_meta, query in query_topic:
        q_dist = list()

        for d_meta, doc in doc_topic:
            distance = distance_measure(query, doc)
            q_dist.append((d_meta, 1.0 - distance))

        ranks.append((q_meta, sorted(q_dist, reverse=True, key=lambda x: x[1])))

    logger.info('Returning %d ranks', len(ranks))
    return ranks


def get_topics(model, corpus):
    logger.info('Getting doc topic for corpus with length %d', len(corpus))
    doc_topic = list()
    corpus.metadata = True

    for doc, metadata in corpus:
        topics = model.__getitem__(doc, eps=0)
        topics = [val for id, val in topics]
        doc_topic.append((metadata, list(sorted(topics, reverse=True))))

    corpus.metadata = False
    logger.info('Returning doc topic of length %d', len(doc_topic))

    return doc_topic

def create_queries(project, id2word):
    corpus_fname_base = project.data_path + 'Queries'
    corpus_fname = corpus_fname_base + '.mallet.gz'

    if not os.path.exists(corpus_fname):
        pp = GeneralCorpus(lazy_dict=True) # configure query preprocessing here

        with open(project.data_path + 'ids.txt') as f:
            ids = [x.strip() for x in f.readlines()]

        queries = list()
        for id in ids:
            with open(project.data_path + 'Queries/ShortDescription' + id + '.txt') as f:
                short = f.read()

            with open(project.data_path + 'Queries/LongDescription' + id + '.txt') as f:
                long = f.read()

            text = ' '.join([short, long])
            text = pp.preprocess(text)

            # this step will remove any words not found in the dictionary
            bow = id2word.doc2bow(text)

            queries.append((bow, (id, 'query')))

        # write the corpus and dictionary to disk. this will take awhile.
        MalletCorpus.serialize(corpus_fname, queries, id2word=id2word,
                                metadata=True)

    # re-open the compressed corpus
    corpus = MalletCorpus(corpus_fname, id2word=id2word)

    return corpus

def load_goldsets(project):
    with open(project.data_path + 'ids.txt') as f:
        ids = [x.strip() for x in f.readlines()]

    goldsets = list()
    for id in ids:
        with open(project.data_path + 'GoldSets/GoldSet' + id + '.txt') as f:
            golds = frozenset(x.strip() for x in f.readlines())

        goldsets.append((id, golds))

    return goldsets



def create_model(project, corpus, id2word, name):
    model_fname = project.data_path + name +  str(project.num_topics) + '.lda'

    if not os.path.exists(model_fname):
        model = LdaModel(corpus,
                         id2word=id2word,
                         alpha=project.alpha,
                         eta=project.eta,
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

    missing_fname = project.data_path + 'missing-gold.txt'
    with open(missing_fname, 'w') as f:
        for each in sorted(list(goldset - taserset)):
            f.write(each + '\n')

    ours_fname = project.data_path + 'allours.txt'
    with open(ours_fname, 'w') as f:
        for each in sorted(list(taserset)):
            f.write(each + '\n')

def create_dictionary(project):
    dict_fname = project.data_path + 'id2word.dict.gz'
    if not os.path.exists(dict_fname):
        id2word = Dictionary()

        # this is required because bool(id2word) => False when empty
        # id2word.add_documents([['seed']])
    else:
        id2word = Dictionary.load(dict_fname)

    return id2word


def create_corpus(project, repos, id2word, Kind):
    corpus_fname_base = project.data_path + Kind.__name__
    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = project.data_path + 'id2word.dict.gz'

    if not os.path.exists(corpus_fname):
        combiner = CorpusCombiner(id2word=id2word)

        for repo in repos:
            try:
                if project.sha:
                    corpus = Kind(repo, project.sha, id2word=id2word, lazy_dict=True)
                else:
                    corpus = Kind(repo, project.ref, id2word=id2word, lazy_dict=True)
            except KeyError:
                continue
            except TaserError:
                continue

            combiner.add(corpus)

        # write the corpus and dictionary to disk. this will take awhile.
        combiner.metadata = True
        MalletCorpus.serialize(corpus_fname, combiner, id2word=id2word,
                               metadata=True)
        combiner.metadata = False

        id2word.save(dict_fname)

    if id2word:
        corpus = MalletCorpus(corpus_fname, id2word=id2word)
    else:
        corpus = MalletCorpus(corpus_fname, id2word=create_dictionary(project))

    return corpus

def create_release_corpus(project, id2word):
    corpus_fname_base = project.data_path + 'TaserReleaseCorpus'
    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = corpus_fname_base + '.dict.gz'

    if not os.path.exists(corpus_fname):
        corpus = TaserReleaseCorpus(project.src_path, id2word=id2word, lazy_dict=True)

        corpus.metadata = True
        MalletCorpus.serialize(corpus_fname, corpus, id2word=id2word,
                               metadata=True)
        corpus.metadata = False
        id2word.save(dict_fname)

    if id2word:
        corpus = MalletCorpus(corpus_fname, id2word=id2word)
    else:
        corpus = MalletCorpus(corpus_fname, id2word=create_dictionary(project))

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
