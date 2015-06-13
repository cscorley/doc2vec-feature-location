#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
logger = logging.getLogger('cfl.main')

import sys
import os.path
import csv
import zipfile
import tarfile
from collections import namedtuple

import click
import dulwich.repo
import scipy.stats
import numpy
from gensim.corpora import MalletCorpus, Dictionary
from gensim.models import LdaModel, LsiModel, Doc2Vec
from gensim.matutils import sparse2full
from gensim.utils import smart_open, to_unicode, to_utf8

import utils
from corpora import (ChangesetCorpus, SnapshotCorpus, ReleaseCorpus,
                     TaserSnapshotCorpus, TaserReleaseCorpus,
                     CorpusCombiner, GeneralCorpus, LabeledCorpus, OrderedCorpus)
from errors import TaserError


def error(*args, **kwargs):
    logger.error(*args)
    if 'errorno' in kwargs:
        sys.exit(kwargs['errorno'])

    sys.exit(1)


@click.command()
@click.option('--verbose', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--force', is_flag=True)
@click.option('--temporal', is_flag=True)
@click.option('--lda', is_flag=True)
@click.option('--lsi', is_flag=True)
@click.option('--d2v', is_flag=True)
@click.option('--name', help="Name of project to run experiment on")
@click.option('--version', help="Version of project to run experiment on")
@click.option('--level', default="class", help="Granularity level of project to run experiment on")
@click.option('--num_topics', default=500, type=click.INT)
@click.option('--chunksize', default=2000, type=click.INT)
@click.option('--passes', default=5, type=click.INT)
@click.option('--iterations', default=1000, type=click.INT)
@click.option('--decay', default=0.5, type=click.FLOAT)
@click.option('--offset', default=1.0, type=click.FLOAT)
@click.option('--eta', type=click.FLOAT)
@click.option('--alpha', default='symmetric')
def cli(debug, verbose, name, version, *args, **kwargs):
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
    projects = load_projects(kwargs)
    for project in projects:
        if name:
            name = name.lower()
            if name == project.name:
                if version and version != project.version:
                    continue

                run_experiment(project)
                sys.exit(0) # done, boom shakalaka
        else:
            run_experiment(project)


def run_experiment(project):
    logger.info("Running project on %s", str(project))
    print(project)

    repos = load_repos(project)

    goldsets = load_goldsets(project)

    # vec relies on words being in order, so lets make a special corpus.
    queries_vec = create_queries_vec(project)
    release_corpus_vec = create_release_corpus_vec(project, repos)

    release_vec = run_basic_vec(project, release_corpus_vec,
                                release_corpus_vec, queries_vec, goldsets,
                                'Release', use_level=True)

    release_vec_sums = run_basic_vec_sums(project, release_corpus_vec,
                                          release_corpus_vec, queries_vec,
                                          goldsets, 'Release', use_level=True)


    queries = create_queries(project)
    release_corpus = create_release_corpus(project, repos)
    release_lda = run_basic(project, release_corpus, release_corpus, queries,
                            goldsets, 'Release', use_level=True)

    do_science('release_lda', release_lda,
            'release_vec', release_vec)

    do_science('release_vec', release_vec,
            'release_vec_sums', release_vec_sums)

def write_ranks(project, prefix, ranks):
    with smart_open(os.path.join(project.full_path, '-'.join([prefix, project.level, str(project.num_topics), 'ranks.csv.gz'])), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'rank', 'distance', 'item'])

        for gid, rank in ranks.items():
            for idx, dist, d_name in rank:
                writer.writerow([gid, idx, dist, to_utf8(d_name)])

def read_ranks(project, prefix):
    ranks = dict()
    with smart_open(os.path.join(project.full_path, '-'.join([prefix, project.level, str(project.num_topics), 'ranks.csv.gz']))) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for g_id, idx, dist, d_name in reader:
            if g_id not in ranks:
                ranks[g_id] = list()

            ranks[g_id].append((int(idx), float(dist), to_unicode(d_name)))

    return ranks

def run_basic_vec_sums(project, corpus, other_corpus, queries, goldsets, kind, use_level=False):
    vec_first_rels_sums = list()
    if project.d2v:
        try:
            vec_ranks_sums = read_ranks(project, kind.lower() + '_vec_sums')
            logger.info("Sucessfully read previously written %s vec ranks", kind)
            exists = True
        except IOError:
            exists = False

        if project.force or not exists:
            vec_model, _ = create_vec_model(project, corpus, kind, use_level=use_level)
            vec_ranks_sums = get_rank_vec(vec_model, queries, goldsets, other_corpus)
            write_ranks(project, kind.lower() + '_vec_sums', vec_ranks_sums)

        vec_first_rels_sums = get_frms(vec_ranks_sums)

    return vec_first_rels_sums


def run_basic_vec(project, corpus, other_corpus, queries, goldsets, kind, use_level=False):
    vec_first_rels = list()
    if project.d2v:
        try:
            vec_ranks = read_ranks(project, kind.lower() + '_vec')
            logger.info("Sucessfully read previously written %s vec ranks", kind)
            exists = True
        except IOError:
            exists = False

        if project.force or not exists:
            vec_model, _ = create_vec_model(project, corpus, kind, use_level=use_level)

            vec_query_topic = get_topics_vec(vec_model, queries, try_model=False)
            vec_doc_topic = get_topics_vec(vec_model, other_corpus)
            vec_ranks = get_rank(goldsets, vec_query_topic, vec_doc_topic)
            write_ranks(project, kind.lower() + '_vec', vec_ranks)

        vec_first_rels = get_frms(vec_ranks)

    return vec_first_rels

def run_basic(project, corpus, other_corpus, queries, goldsets, kind, use_level=False):
    """
    This function runs the experiment in one-shot. It does not evaluate the
    changesets over time.
    """
    logger.info("Running basic evaluation on the %s", kind)


    lda_first_rels = list()
    if project.lda:
        try:
            lda_ranks = read_ranks(project, kind.lower() + '_lda')
            logger.info("Sucessfully read previously written %s LDA ranks", kind)
            exists = True
        except IOError:
            exists = False

        if project.force or not exists:
            lda_model, _ = create_lda_model(project, corpus, corpus.id2word, kind, use_level=use_level)
            lda_query_topic = get_topics(lda_model, queries)
            lda_doc_topic = get_topics(lda_model, other_corpus)

            lda_ranks = get_rank(goldsets, lda_query_topic, lda_doc_topic)
            write_ranks(project, kind.lower() + '_lda', lda_ranks)

        lda_first_rels = get_frms(lda_ranks)

    return lda_first_rels

def run_temporal(project, repos, corpus, queries, goldsets, force=False):
    logger.info("Running temporal evaluation")

    try:
        lda_ranks = read_ranks(project, 'temporal')
        lda_rels = get_frms(lda_ranks)

        vec_ranks = read_ranks(project, 'temporal_vec')
        vec_rels = get_frms(vec_ranks)
        logger.info("Sucessfully read previously written Temporal ranks")
    except IOError:
        force = True

    if force:
        lda_rels, vec_rels = run_temporal_helper(project, repos, corpus, queries, goldsets)

    return lda_rels, vec_rels


def run_temporal_helper(project, repos, corpus, queries, goldsets):
    """
    This function runs the experiment in over time. That is, it stops whenever
    it reaches a commit linked with an issue/query. Will not work on all
    projects.
    """
    ids = set(i for i,g in goldsets.items())
    issue2git, git2issue = load_issue2git(project, ids)

    logger.info("Stopping at %d commits for %d issues", len(git2issue), len(issue2git))

    lda, lda_fname = create_lda_model(project, None, corpus.id2word,
                                      'Temporal', use_level=False, force=True)

    vec, vec_fname = create_vec_model(project, None, corpus.id2word,
                                      'Temporal', use_level=False, force=True)

    indices = list()
    lda_ranks = dict()
    vec_ranks = dict()
    docs = list()
    corpus.metadata = True
    prev = 0

    # let's partition the corpus first
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
        for i in xrange(start, end):
            docs.append(corpus[i])

        lda.update(docs, decay=0.99)
        vec.add_documents(docs)

        for qid in git2issue[sha]:
            logger.info('Getting ranks for query id %s', qid)
            # build a snapshot corpus of items *at this commit*
            try:
                other_corpus = create_release_corpus(project, repos, forced_ref=sha)
            except TaserError:
                continue

            # do LDA magic
            lda_query_topic = get_topics(lda, queries, by_ids=[qid])
            lda_doc_topic = get_topics(lda, other_corpus)
            lda_subranks = get_rank(goldsets, lda_query_topic, lda_doc_topic)
            if qid in lda_subranks:
                if qid not in lda_ranks:
                    lda_ranks[qid] = list()

                rank = lda_subranks[qid]
                lda_ranks[qid].extend(rank)
            else:
                logger.info('Couldnt find qid %s', qid)

            # do vec magic
            vec_subranks = get_rank_vec(vec, queries, goldsets, other_corpus, by_ids=[qid])
            if qid in vec_subranks:
                if qid not in vec_ranks:
                    vec_ranks[qid] = list()

                rank = vec_subranks[qid]
                vec_ranks[qid].extend(rank)
            else:
                logger.info('Couldnt find qid %s', qid)

    lda.save(lda_fname)
    vec.save(vec_fname)

    write_ranks(project, 'temporal', lda_ranks)
    write_ranks(project, 'temporal_vec', vec_ranks)

    lda_rels = get_frms(lda_ranks)
    vec_rels = get_frms(vec_ranks)

    return lda_rels, vec_rels


def merge_first_rels(a, b, ignore=False):
    logger.info("Merging %d, %d", len(a), len(b))
    first_rels = dict()

    for num, query_id, doc_meta in a:
        qid = int(query_id)
        if qid not in first_rels:
            first_rels[qid] = [num]
        else:
            logger.info('duplicate qid found: %s', query_id)

    for num, query_id, doc_meta in b:
        qid = int(query_id)
        if qid not in first_rels and not ignore:
            logger.info('qid not found: %s', qid)
            first_rels[qid] = [0]

        if qid in first_rels:
            first_rels[qid].append(num)

    for key, v in first_rels.items():
        if len(v) == 1:
            v.append(0)

    x = [v[0] for v in first_rels.values()]
    y = [v[1] for v in first_rels.values()]
    logger.info("Returning merged %d, %d", len(x), len(y))
    return x, y

def do_science(prefixa, relsa, prefixb, relsb, ignore=False):
    # Build a dictionary with each of the results for stats.

    x, y = merge_first_rels(relsa, relsb, ignore=ignore)



    print(prefixa+' mrr:', utils.calculate_mrr(x))
    print(prefixb+' mrr:', utils.calculate_mrr(y))
    print('wilcoxon signedrank:', scipy.stats.wilcoxon(x, y))

    #print(prefix+' ranksums:', scipy.stats.ranksums(x, y))
    #print(prefix+' mann-whitney:', scipy.stats.mannwhitneyu(x, y))
    #print('friedman:', scipy.stats.friedmanchisquare(x, y, x2, y2))


def get_frms(ranks):
    logger.info('Getting FRMS for %d ranks', len(ranks))
    frms = list()

    for r_id, rank in ranks.items():
        if rank:
            idx, dist, meta = rank[0]
            frms.append((idx, r_id, meta))

    logger.info('Returning %d FRMS', len(frms))
    return frms

def get_rels(goldset, ranks):
    rels = list()

    for idx, rank in enumerate(ranks):
        dist, meta = rank
        d_name, d_repo = meta
        if d_name in goldset:
            rels.append((idx+1, dist, d_name))

    rels.sort()

    return rels

def get_rank_vec(model, queries, goldsets, corpus, by_ids=None):
    queries = LabeledCorpus(queries.fname)
    corpus = LabeledCorpus(corpus.fname)
    logger.info('Getting ranks for Doc2Vec model')

    ranks = dict()
    for query in queries:
        q_dist = list()

        # labeledcorpus is arhghhh
        qid = query.tags[0].replace('DOC__', '')
        if by_ids is not None and qid not in by_ids:
            logger.info('skipping')
            continue

        qwords = filter(lambda x: x in model.vocab, query.words)

        for doc in corpus:
            did = doc.tags[0].replace('DOC__', '')
            dwords = filter(lambda x: x in model.vocab, doc.words)

            if len(dwords) and len(qwords):
                # best thing to do without inference
                sim = model.n_similarity(qwords, dwords)
                q_dist.append((1.0 - sim, (did, 'UNKNOWN')))

        q_dist.sort()
        if qid in goldsets:
            goldset = goldsets[qid]
            ranks[qid] = get_rels(goldset, q_dist)
        else:
            logger.info("Could not find goldset for query %s", qid)

    return ranks


def get_rank(goldsets, query_topic, doc_topic, distance_measure=utils.cosine_distance):
    logger.info('Getting ranks between %d query topics and %d doc topics',
                len(query_topic), len(doc_topic))
    ranks = dict()
    for q_meta, query in query_topic:
        qid, _ = q_meta
        q_dist = list()

        for d_meta, doc in doc_topic:
            distance = distance_measure(query, doc)
            q_dist.append((distance, d_meta))

        q_dist.sort()
        if qid in goldsets:
            goldset = goldsets[qid]
            ranks[qid] = get_rels(goldset, q_dist)
        else:
            logger.info("Could not find goldset for query %s", qid)

    logger.info('Returning %d ranks', len(ranks))
    return ranks


def get_topics(model, corpus, by_ids=None, full=True):
    logger.info('Getting doc topic for corpus with length %d', len(corpus))
    doc_topic = list()
    corpus.metadata = True
    old_id2word = corpus.id2word
    corpus.id2word = model.id2word

    for doc, metadata in corpus:
        if by_ids is None or metadata[0] in by_ids:
            # get a vector where low topic values are zeroed out.
            topics = model[doc]
            if full:
                topics = sparse2full(topics, model.num_topics)

            # this gets the "full" vector that includes low topic values
            # topics = model.__getitem__(doc, eps=0)
            # topics = [val for id, val in topics]

            doc_topic.append((metadata, topics))

    corpus.metadata = False
    corpus.id2word = old_id2word
    logger.info('Returning doc topic of length %d', len(doc_topic))

    return doc_topic

def get_topics_vec(model, corpus, by_ids=None, full=True, try_model=True):
    logger.info('Getting doc topic for corpus with length %d', len(corpus))
    doc_topic = list()
    corpus.metadata = True

    for doc, metadata in corpus:
        # lol gensim
        if by_ids is None or metadata[0] in by_ids:
            try:
                if not try_model:
                    raise KeyError
                topics = model.docvecs['DOC__' + metadata[0]]
            except KeyError:
                topics = model.infer_vector(doc)

            doc_topic.append((metadata, topics))

    corpus.metadata = False
    logger.info('Returning doc topic of length %d', len(doc_topic))

    return doc_topic


def load_goldsets(project):
    logger.info("Loading goldsets for project: %s", str(project))
    with open(os.path.join(project.full_path, 'ids.txt')) as f:
        ids = [x.strip() for x in f.readlines()]

    goldsets = dict()
    s = 0
    for id in ids:
        try:
            with open(os.path.join(project.full_path, 'goldsets', project.level,
                                    id + '.txt')) as f:
                golds = frozenset(x.strip() for x in f.readlines())
                s += len(golds)

            goldsets[id] = golds
        except IOError:
            pass

    logger.info("Returning %d goldsets %d", len(goldsets), s)
    return goldsets


def load_issue2git(project, ids):
    dest_fn = os.path.join(project.full_path, 'issue2git.csv')
    if os.path.exists(dest_fn):
        write_out = False
        i2g = dict()
        with open(dest_fn) as f:
            r = csv.reader(f)
            for row in r:
                i2g[row[0]] = row[1:]
    else:
        write_out = True

        i2s = dict()
        fn = os.path.join(project.full_path, 'IssuesToSVNCommitsMapping.txt')
        with open(fn) as f:
            lines = [line.strip().split('\t') for line in f]
            for line in lines:
                issue = line[0]
                links = line[1]
                svns = line[2:]

                i2s[issue] = svns

        s2g = dict()
        fn = os.path.join(project.data_path, 'svn2git.csv')
        with open(fn) as f:
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

    # Make sure we have a commit for all issues
    keys = set(i2g.keys())
    ignore = ids - keys
    if len(ignore):
        logger.info("Ignoring evaluation for the following issues:\n\t%s",
                    '\n\t'.join(ignore))

    # clean up issue2git
    for issue in i2g.keys():
        if issue in ignore or issue not in ids:
            i2g.pop(issue)

    # build reverse mapping
    g2i = dict()
    for issue, gits in i2g.items():
        for git in gits:
            if git not in g2i:
                g2i[git] = list()
            g2i[git].append(issue)

    if write_out:
        with open(dest_fn, 'w') as f:
            w = csv.writer(f)
            for issue, gits in i2g.items():
                w.writerow([issue] + gits)

    return i2g, g2i


def load_projects(config):
    projects = list()
    with open("projects.csv", 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        customs = ['data_path', 'full_path', 'src_path']
        Project = namedtuple('Project',  ' '.join(header + customs + config.keys()))
        # figure out which column index contains the project name
        name_idx = header.index("name")
        version_idx = header.index("version")

        # find the project in the csv, adding it's info to config
        for row in reader:
            # built the data_path value
            row += (os.path.join('data', row[name_idx], ''),)

            # build the full_path value
            row += (os.path.join('data', row[name_idx], row[version_idx], ''),)

            # build the src_path value
            row += (os.path.join('data', row[name_idx], row[version_idx], 'src'),)

            row += config.values()

            projects.append(Project(*row))

    return projects


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
            repo = utils.clone(url, target, bare=True)
        except OSError:
            repo = dulwich.repo.Repo(target)

        repos.append(repo)

    return repos


def create_queries(project):
    corpus_fname_base = project.full_path + 'Queries'
    corpus_fname = corpus_fname_base + '.ordered.gz'
    dict_fname = corpus_fname_base + '.dict.gz'

    if not os.path.exists(corpus_fname):
        pp = GeneralCorpus(lazy_dict=True)
        id2word = Dictionary()

        with open(os.path.join(project.full_path, 'ids.txt')) as f:
            ids = [x.strip() for x in f.readlines()]

        queries = list()
        for id in ids:
            with open(os.path.join(project.full_path, 'queries',
                                    'ShortDescription' + id + '.txt')) as f:
                short = f.read()

            with open(os.path.join(project.full_path, 'queries',
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

def create_queries_vec(project):
    corpus_fname_base = project.full_path + 'Queries'
    corpus_fname = corpus_fname_base + '.ordered.gz'
    dict_fname = corpus_fname_base + '.dict.gz'

    if not os.path.exists(corpus_fname):
        pp = GeneralCorpus(lazy_dict=True)
        id2word = Dictionary()

        with open(os.path.join(project.full_path, 'ids.txt')) as f:
            ids = [x.strip() for x in f.readlines()]

        queries = list()
        for id in ids:
            with open(os.path.join(project.full_path, 'queries',
                                    'ShortDescription' + id + '.txt')) as f:
                short = f.read()

            with open(os.path.join(project.full_path, 'queries',
                                    'LongDescription' + id + '.txt')) as f:
                long = f.read()

            text = ' '.join([short, long])
            text = list(pp.preprocess(text))

            queries.append((text, (id, 'query')))

        OrderedCorpus.serialize(corpus_fname, queries, metadata=True)

    corpus = OrderedCorpus(corpus_fname)
    return corpus

def create_lda_model(project, corpus, id2word, name, use_level=True, force=False):
    model_fname = project.full_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.lda.gz'


    if not os.path.exists(model_fname) or project.force or force:
        if corpus:
            update_every=None # run in batch if we have a pre-supplied corpus
        else:
            update_every=1

        model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         alpha=project.alpha,
                         eta=project.eta,
                         chunksize=project.chunksize,
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

def create_lsi_model(project, corpus, id2word, name, use_level=True, force=False):
    model_fname = project.full_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.lsi.gz'

    if not os.path.exists(model_fname) or force:
        model = LsiModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=project.num_topics,
                         )

        if corpus:
            model.save(model_fname)
    else:
        model = LsiModel.load(model_fname)

    return model, model_fname

def create_vec_model(project, corpus, name, use_level=True, force=False):
    model_fname = project.full_path + name + str(project.num_topics)
    if use_level:
        model_fname += project.level

    model_fname += '.vec.gz'

    corpus = LabeledCorpus(corpus.fname)

    if not os.path.exists(model_fname) or project.force or force:
        model = Doc2Vec(corpus, min_count=1, size=project.num_topics)

        if corpus:
            model.save(model_fname)
    else:
        model = Doc2Vec.load(model_fname)

    return model, model_fname



def create_mallet_model(project, corpus, name, use_level=True):
    model_fname = project.full_path + name + str(project.num_topics)
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
    corpus_fname_base = project.full_path + Kind.__name__

    if use_level:
        corpus_fname_base += project.level

    if forced_ref:
        corpus_fname_base += forced_ref[:8]

    corpus_fname = corpus_fname_base + '.ordered.gz'
    dict_fname = corpus_fname_base + '.dict.gz'
    made_one = False

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
            except TaserError as e:
                if repo == repos[-1] and not made_one:
                    raise e
                    # basically, if we are at the last repo and we STILL
                    # haven't sucessfully extracted a corpus, ring some bells
                else:
                    # otherwise, keep trying. winners never quit.
                    continue

            combiner.add(corpus)
            made_one = True

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

    return create_corpus(project, repos, SC, forced_ref=forced_ref)

    # not using this just yet
    if forced_ref:
        return create_corpus(project, repos, SC, forced_ref=forced_ref)
    else:
        try:
            return create_corpus(project, [None], RC)
        except TaserError:
            return create_corpus(project, repos, SC, forced_ref=forced_ref)


def create_release_corpus_vec(project, repos, forced_ref=None):
    if project.level == 'file':
        RC = ReleaseCorpus
        SC = SnapshotCorpus
    else:
        RC = TaserReleaseCorpus
        SC = TaserSnapshotCorpus

    return create_corpus_vec(project, repos, SC, forced_ref=forced_ref)

    # not using this just yet
    if forced_ref:
        return create_corpus_vec(project, repos, SC, forced_ref=forced_ref)
    else:
        try:
            return create_corpus_vec(project, [None], RC)
        except TaserError:
            return create_corpus_vec(project, repos, SC, forced_ref=forced_ref)


def create_corpus_vec(project, repos, Kind, use_level=True, forced_ref=None):
    corpus_fname_base = project.full_path + Kind.__name__

    if use_level:
        corpus_fname_base += project.level

    if forced_ref:
        corpus_fname_base += forced_ref[:8]

    corpus_fname = corpus_fname_base + '.ordered.gz'
    dict_fname = corpus_fname_base + '.dict.gz'
    made_one = False

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
            except TaserError as e:
                if repo == repos[-1] and not made_one:
                    raise e
                    # basically, if we are at the last repo and we STILL
                    # haven't sucessfully extracted a corpus, ring some bells
                else:
                    # otherwise, keep trying. winners never quit.
                    continue

            combiner.add(corpus)
            made_one = True

        # write the corpus and dictionary to disk. this will take awhile.
        combiner.metadata = True
        OrderedCorpus.serialize(corpus_fname, combiner, metadata=True)
        combiner.metadata = False

    corpus = OrderedCorpus(corpus_fname)

    return corpus

