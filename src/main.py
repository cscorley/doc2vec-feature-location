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

import click
import dulwich.client
import dulwich.repo
import scipy.stats
import numpy
from gensim.corpora import MalletCorpus, Dictionary
from gensim.models import LdaModel, LsiModel, LdaMallet
from gensim.matutils import sparse2full

import utils
from corpora import (ChangesetCorpus, SnapshotCorpus, ReleaseCorpus,
                     TaserSnapshotCorpus, TaserReleaseCorpus,
                     CorpusCombiner, GeneralCorpus)
from errors import TaserError


def error(*args, **kwargs):
    logger.error(*args)
    if 'errorno' in kwargs:
        sys.exit(kwargs['errorno'])

    sys.exit(1)


@click.command()
@click.option('--verbose', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--path', default='data/', help="Set the directory to work within")
@click.argument('name')
@click.option('--version', help="Version of project to run experiment on")
@click.option('--level', help="Granularity level of project to run experiment on")
def cli(verbose, debug, path, name, version, level):
    """
    Changesets for Feature Location
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : ' +
                        '%(name)s : %(funcName)s : %(message)s')

    if debug:
        logging.root.setLevel(level=logging.DEBUG)
    elif verbose:
        logging.root.setLevel(level=logging.INFO)
    else:
        logging.root.setLevel(level=logging.ERROR)

    # load project info
    project = load_project(name, version, level)
    logger.info("Running project on %s", str(project))
    repos = load_repos(project)

    # create/load document lists
    queries = create_queries(project)
    goldsets = load_goldsets(project)

    # get corpora
    changeset_corpus = create_corpus(project, repos, ChangesetCorpus, use_level=False)
    release_corpus = create_release_corpus(project, repos)

    # release-based evaluation is #basic 💁
    release_lda, release_lsi = run_basic(project, release_corpus,
                                         release_corpus, queries, goldsets,
                                         'Release', use_level=True)

    write_ranks(project, 'release_lda', release_lda)
    write_ranks(project, 'release_lsi', release_lsi)

    changeset_lda, changeset_lsi = run_basic(project, changeset_corpus,
                                             release_corpus, queries, goldsets,
                                             'Changeset')

    write_ranks(project, 'changeset_lda', changeset_lda)
    write_ranks(project, 'changeset_lsi', changeset_lsi)

    try:
        temporal_lda, temporal_lsi = run_temporal(project, repos,
                                                  changeset_corpus, queries,
                                                  goldsets)
    except IOError:
        logger.info("Files needed for temporal evaluation not found. Skipping.")
    else:
        write_ranks(project, 'temporal_lda', temporal_lda)
        write_ranks(project, 'temporal_lsi', temporal_lsi)
        do_science('temporal_lda', temporal_lda, release_lda)
        do_science('temporal_lsi', temporal_lsi, release_lsi)

    # do this last so that the results are printed together
    do_science('basic_lda', changeset_lda, release_lda)
    do_science('basic_lsi', changeset_lsi, release_lsi)

def write_ranks(project, prefix, ranks):
    with open(os.path.join(project.data_path, prefix + '-ranks.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(ranks)


def run_basic(project, corpus, other_corpus, queries, goldsets, kind, use_level=False):
    """
    This function runs the experiment in one-shot. It does not evaluate the
    changesets over time.
    """
    logger.info("Running basic evaluation on the %s", kind)
    lda_model, _ = create_lda_model(project, corpus, corpus.id2word, kind, use_level=use_level)
    lda_query_topic = get_topics(lda_model, queries)
    lda_doc_topic = get_topics(lda_model, other_corpus)
    lda_ranks = get_rank(lda_query_topic, lda_doc_topic)
    lda_first_rels = get_frms(goldsets, lda_ranks)

    lsi_model, _ = create_lsi_model(project, corpus, corpus.id2word, kind, use_level=use_level)
    lsi_query_topic = get_topics(lsi_model, queries)
    lsi_doc_topic = get_topics(lsi_model, other_corpus)

    # for some reason the ranks from LSI cause hellinger_distance to cry
    lsi_ranks = get_rank(lsi_query_topic, lsi_doc_topic, utils.cosine_distance)
    lsi_first_rels = get_frms(goldsets, lsi_ranks)

    return lda_first_rels, lsi_first_rels


def run_temporal(project, repos, corpus, queries, goldsets):
    """
    This function runs the experiment in over time. That is, it stops whenever
    it reaches a commit linked with an issue/query. Will not work on all
    projects.
    """
    logger.info("Running temporal evaluation")

    issue2git, git2issue = load_issue2git(project)

    logger.info("Stopping at %d commits for %d issues", len(git2issue), len(issue2git))

    # Make sure we have a commit for all issues
    ids = set(i for i,g in goldsets)
    keys = set(issue2git.keys())
    ignore = ids - keys
    if len(ignore):
        logger.info("Ignoring evaluation for the following issues:\n\t%s",
                    '\n\t'.join(ignore))


    lda, lda_fname = create_lda_model(project, None, corpus.id2word, 'Temporal',
                                      use_level=False, load=False)

    lsi, lsi_fname = create_lsi_model(project, None, corpus.id2word, 'Temporal',
                                      use_level=False, load=False)

    indices = list()
    lda_ranks = dict()
    lsi_ranks = dict()
    docs = list()
    corpus.metadata = True
    prev = 0

    # let's partition the corpus first?
    for idx, docmeta in enumerate(corpus):
        doc, meta = docmeta
        sha, _ = meta
        if sha in git2issue:
            indices.append((prev, idx+1, sha))
            prev = idx

    logger.info('Created %d partitions of the corpus', len(indices))
    corpus.metadata = False

    for counter, index  in enumerate(indices):
        logger.info('At %d of %d partitions', counter, len(indices))
        start, end, sha = index
        docs = list()
        for i in range(start, end):
            docs.append(corpus[i])

        lda.update(docs, decay=2.0)
        lsi.add_documents(docs)

        for qid in git2issue[sha]:
            if qid not in ignore:
                logger.info('Getting ranks for query id %s', qid)
                # build a snapshot corpus of items *at this commit*
                other_corpus = create_release_corpus(project, repos, forced_ref=sha)

                # do LDA magic
                lda_query_topic = get_topics(lda, queries, by_id=qid)
                lda_doc_topic = get_topics(lda, other_corpus)
                lda_subranks = get_rank(lda_query_topic, lda_doc_topic)
                if qid in lda_subranks:
                    if qid not in lda_ranks:
                        lda_ranks[qid] = list()

                    rank = lda_subranks[qid]
                    lda_ranks[qid].extend(rank)
                else:
                    logger.info('Couldnt find qid %s', qid)

                # do LSI magic
                lsi_query_topic = get_topics(lsi, queries, by_id=qid)
                lsi_doc_topic = get_topics(lsi, other_corpus)
                # for some reason the ranks from LSI cause hellinger_distance to
                # cry, use cosine distance instead
                lsi_subranks = get_rank(lsi_query_topic, lsi_doc_topic, utils.cosine_distance)
                if qid in lsi_subranks:
                    if qid not in lsi_ranks:
                        lsi_ranks[qid] = list()

                    rank = lsi_subranks[qid]
                    lsi_ranks[qid].extend(rank)
                else:
                    logger.info('Couldnt find qid %s', qid)

    lda.save(lda_fname)
    lsi.save(lsi_fname)

    lda_rels = get_frms(goldsets, lda_ranks)
    lsi_rels = get_frms(goldsets, lsi_ranks)
    return lda_rels, lsi_rels



def do_science(prefix, changeset_first_rels, release_first_rels):
    # Build a dictionary with each of the results for stats.
    first_rels = dict()

    for num, query_id, doc_meta in changeset_first_rels:
        if query_id not in first_rels:
            first_rels[query_id] = [num]
        else:
            logger.info('duplicate qid found: %s', query_id)

    for num, query_id, doc_meta in release_first_rels:
        if query_id not in first_rels:
            logger.info('qid not found: %s', query_id)
            first_rels[query_id] = [0]

        first_rels[query_id].append(num)

    for key, v in first_rels.items():
        if len(v) == 1:
            v.append(0)

    x = [v[0] for v in first_rels.values()]
    y = [v[1] for v in first_rels.values()]

    print(prefix+'changeset mrr:', utils.calculate_mrr(x))
    print(prefix+'release mrr:', utils.calculate_mrr(y))
    print(prefix+'ranksums:', scipy.stats.ranksums(x, y))
    print(prefix+'mann-whitney:', scipy.stats.mannwhitneyu(x, y))
    print(prefix+'wilcoxon signedrank:', scipy.stats.wilcoxon(x, y))
    #print('friedman:', scipy.stats.friedmanchisquare(x, y, x2, y2))


def get_frms(goldsets, ranks):
    logger.info('Getting FRMS for %d goldsets and %d ranks',
                len(goldsets), len(ranks))
    frms = list()

    for g_id, goldset in goldsets:
        if g_id not in ranks:
            logger.info('Could not find ranks for goldset id %s', g_id)
        else:
            logger.info('Getting best rank out of %d shas', len(ranks[g_id]))
            subfrms = list()
            for rank in ranks[g_id]:
                for idx, metadist in enumerate(rank):
                    doc_meta, dist = metadist
                    d_name, d_repo = doc_meta
                    if d_name in goldset:
                        subfrms.append((idx+1, g_id, doc_meta))
                        break

            # i think this might be cheating? :)
            subfrms.sort()
            logger.debug(str(subfrms))
            if subfrms:
                frms.append(subfrms[0])


    logger.info('Returning %d FRMS', len(frms))
    return frms


def get_rank(query_topic, doc_topic, distance_measure=utils.hellinger_distance):
    logger.info('Getting ranks between %d query topics and %d doc topics',
                len(query_topic), len(doc_topic))
    ranks = dict()
    for q_meta, query in query_topic:
        qid, _ = q_meta
        q_dist = list()

        for d_meta, doc in doc_topic:
            distance = distance_measure(query, doc)
            q_dist.append((d_meta, distance))

        if qid not in ranks:
            ranks[qid] = list()

        ranks[qid].append(list(sorted(q_dist, key=lambda x: x[1])))

    logger.info('Returning %d ranks', len(ranks))
    return ranks


def get_topics(model, corpus, by_id=None):
    logger.info('Getting doc topic for corpus with length %d', len(corpus))
    doc_topic = list()
    corpus.metadata = True
    old_id2word = corpus.id2word
    corpus.id2word = model.id2word

    for doc, metadata in corpus:
        if by_id is None or metadata[0] == by_id:
            # get a vector where low topic values are zeroed out.
            topics = sparse2full(model[doc], model.num_topics)

            # this gets the "full" vector that includes low topic values
            # topics = model.__getitem__(doc, eps=0)
            # topics = [val for id, val in topics]

            doc_topic.append((metadata, topics))

    corpus.metadata = False
    corpus.id2word = old_id2word
    logger.info('Returning doc topic of length %d', len(doc_topic))

    return doc_topic


def load_goldsets(project):
    logger.info("Loading goldsets for project: %s", str(project))
    with open(os.path.join(project.data_path, 'ids.txt')) as f:
        ids = [x.strip() for x in f.readlines()]

    goldsets = list()
    for id in ids:
        with open(os.path.join(project.data_path, 'goldsets', project.level,
                                id + '.txt')) as f:
            golds = frozenset(x.strip() for x in f.readlines())

        goldsets.append((id, golds))

    logger.info("Returning %d goldsets", len(goldsets))
    return goldsets


def load_issue2git(project):
    fn = 'IssuesToSVNCommitsMapping.txt'
    i2s = dict()
    with open(os.path.join(project.data_path, fn)) as f:
        lines = [line.strip().split('\t') for line in f]
        for line in lines:
            issue = line[0]
            links = line[1]
            svns = line[2:]

            i2s[issue] = svns

    fn = 'svn2git.csv'
    s2g = dict()
    with open(os.path.join('data', project.name, fn)) as f:
        reader = csv.reader(f)
        for svn,git in reader:
            if svn in s2g and s2g[svn] != git:
                logger.info('Different gits sha for SVN revision number %s', svn)
            else:
                s2g[svn] = git

    i2g = dict()
    for issue, svns in i2s.items():
        for svn in svns:
            if svn in s2g:
                # make sure we don't have issues that are empty
                if issue not in i2g:
                    i2g[issue] = list()
                i2g[issue].append(s2g[svn])
            else:
                logger.info('Could not find git sha for SVN revision number %s', svn)

    # build reverse mapping
    g2i = dict()
    for issue, gits in i2g.items():
        for git in gits:
            if git not in g2i:
                g2i[git] = list()
            g2i[git].append(issue)

    return i2g, g2i


def load_project(name, version, level):
    project = None
    with open("projects.csv", 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        customs = ['data_path', 'src_path']
        Project = namedtuple('Project',  ' '.join(header + customs))
        # figure out which column index contains the project name
        name_idx = header.index("name")
        version_idx = header.index("version")
        level_idx = header.index("level")

        # find the project in the csv, adding it's info to config
        for row in reader:
            if name == row[name_idx]:

                # if version specified, make sure we are at the correct one
                if version and version != row[version_idx]:
                    continue

                # if level specified, make sure we are at the correct one
                if level and level != row[level_idx]:
                    continue

                # build the data_path value
                row += (os.path.join('data', row[name_idx], row[version_idx],
                                     ''),)

                # build the src_path value
                row += (os.path.join('data', row[name_idx], row[version_idx],
                                     'src'),)

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
                        # set all empty fields to None
                        row[idx] = None

                # 🎶  do you believe in magicccccc
                # in a young girl's heart? 🎶
                project = Project(*row)
                break

        # we can access project info by:
        #    project.name => "Blah Name"
        #    project.version => "0.22"

        if project is None and version:
            error("Could not find '%s %s' in 'projects.csv'!", name, version)
        elif project is None:
            error("Could not find '%s' in 'projects.csv'!", name)

    return project


def load_repos(project):
    # reading in repos
    with open(os.path.join('data', project.name, 'repos.txt')) as f:
        repo_urls = [line.strip() for line in f]

    repos_base = 'gits'
    if not os.path.exists(repos_base):
        utils.mkdir(repos_base)

    repos = list()

    for url in repo_urls:
        repo_name = url.split('/')[-1]
        target = os.path.join(repos_base, repo_name)
        try:
            repo = clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        repos.append(repo)

    return repos


def create_queries(project):
    corpus_fname_base = project.data_path + 'Queries'
    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = corpus_fname_base + '.dict.gz'

    if not os.path.exists(corpus_fname):
        pp = GeneralCorpus(lazy_dict=True)
        id2word = Dictionary()

        with open(os.path.join(project.data_path, 'ids.txt')) as f:
            ids = [x.strip() for x in f.readlines()]

        queries = list()
        for id in ids:
            with open(os.path.join(project.data_path, 'queries',
                                    'ShortDescription' + id + '.txt')) as f:
                short = f.read()

            with open(os.path.join(project.data_path, 'queries',
                                    'LongDescription' + id + '.txt')) as f:
                long = f.read()

            text = ' '.join([short, long])
            text = pp.preprocess(text)

            # this step will remove any words not found in the dictionary
            bow = id2word.doc2bow(text, allow_update=True)

            queries.append((bow, (id, 'query')))

        # write the corpus and dictionary to disk. this will take awhile.
        MalletCorpus.serialize(corpus_fname, queries, id2word=id2word,
                               metadata=True)

    # re-open the compressed versions of the dictionary and corpus
    id2word = None
    if os.path.exists(dict_fname):
        id2word = Dictionary.load(dict_fname)

    corpus = MalletCorpus(corpus_fname, id2word=id2word)

    return corpus


def create_lda_model(project, corpus, id2word, name, use_level=True, load=True):
    model_fname = project.data_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.lda.gz'


    if not os.path.exists(model_fname) or not load:
        if corpus:
            update_every=None # run in batch if we have a pre-supplied corpus
        else:
            update_every=1

        model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         alpha=project.alpha,
                         eta=project.eta,
                         passes=project.passes,
                         num_topics=project.num_topics,
                         iterations=project.iterations,
                         eval_every=None, # disable perplexity tests for speed
                         update_every=update_every,
                         )

        if corpus:
            model.save(model_fname)
    else:
        model = LdaModel.load(model_fname)

    return model, model_fname

def create_lsi_model(project, corpus, id2word, name, use_level=True, load=True):
    model_fname = project.data_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.lsi.gz'

    if not os.path.exists(model_fname) or not load:
        model = LsiModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=project.num_topics,
                         )

        if corpus:
            model.save(model_fname)
    else:
        model = LsiModel.load(model_fname)

    return model, model_fname


def create_mallet_model(project, corpus, name, use_level=True):
    model_fname = project.data_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.malletlda.gz'

    if not os.path.exists(model_fname):
        model = LdaMallet('./lib/mallet-2.0.7/bin/mallet',
                          corpus=corpus,
                          id2word=corpus.id2word,
                          optimize_interval=10,
                          num_topics=project.num_topics)

        model.save(model_fname)
    else:
        model = LdaMallet.load(model_fname)

    return model


def create_corpus(project, repos, Kind, use_level=True, forced_ref=None):
    corpus_fname_base = project.data_path + Kind.__name__

    if use_level:
        corpus_fname_base += project.level

    if forced_ref:
        corpus_fname_base += forced_ref[:8]

    corpus_fname = corpus_fname_base + '.mallet.gz'
    dict_fname = corpus_fname_base + '.dict.gz'

    if not os.path.exists(corpus_fname):
        combiner = CorpusCombiner()

        for repo in repos:
            try:
                if repo or forced_ref:
                    corpus = Kind(project=project,
                                  repo=repo,
                                  lazy_dict=True,
                                  ref=forced_ref,
                                  )
                else:
                    corpus = Kind(project=project, lazy_dict=True)

            except KeyError:
                continue

            combiner.add(corpus)

        # write the corpus and dictionary to disk. this will take awhile.
        combiner.metadata = True
        MalletCorpus.serialize(corpus_fname, combiner, id2word=combiner.id2word,
                               metadata=True)
        combiner.metadata = False

        # write out the dictionary
        combiner.id2word.save(dict_fname)

    # re-open the compressed versions of the dictionary and corpus
    id2word = None
    if os.path.exists(dict_fname):
        id2word = Dictionary.load(dict_fname)

    corpus = MalletCorpus(corpus_fname, id2word=id2word)

    return corpus


def create_release_corpus(project, repos, forced_ref=None):
    if project.level == 'file':
        RC = ReleaseCorpus
        SC = SnapshotCorpus
    else:
        RC = TaserReleaseCorpus
        SC = TaserSnapshotCorpus

    if forced_ref:
        return create_corpus(project, repos, SC, forced_ref=forced_ref)
    else:
        try:
            return create_corpus(project, [None], RC)
        except TaserError:
            return create_corpus(project, repos, SC, forced_ref=forced_ref)


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


def write_out_missing(project, all_taser):
    goldset = set()
    all_taser.metadata = True
    taserset = set(doc[1][0] for doc in all_taser)
    all_taser.metadata = False

    goldset_fname = project.data_path + 'allgold.txt'
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
