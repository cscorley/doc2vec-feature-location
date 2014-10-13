#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

if __name__ == '__main__':
    import nose
    nose.main()

import unittest
import tempfile
import os
import os.path
from io import StringIO
from collections import namedtuple

from nose.tools import *
import dulwich.repo
from gensim.corpora import MalletCorpus

from src.corpora import (SnapshotCorpus, ReleaseCorpus, ChangesetCorpus,
                         CorpusCombiner)

# datapath is now a useful function for building paths to test files
module_path = os.path.dirname(__file__)
datapath = lambda fname: os.path.join(module_path, u'test_data', fname)

class TestGitCorpus(unittest.TestCase):
    def setUp(self):
        self.Project = namedtuple('Project', 'ref level src_path')
        self.basepath = datapath(u'test_git/')
        if not os.path.exists(self.basepath):
            extraction_path = datapath('')
            gz = datapath(u'test_git.tar.gz')

            import tarfile
            with tarfile.open(gz) as tar:
                tar.extractall(extraction_path)

        self.repo = dulwich.repo.Repo(self.basepath)
        _, self.tempfname = tempfile.mkstemp(text=True)

    def tearDown(self):
        os.remove(self.tempfname)


class TestSnapshotCorpus(TestGitCorpus):
    def setUp(self):
        super(TestSnapshotCorpus, self).setUp()
        self.corpus = SnapshotCorpus(repo=self.repo,
                                     remove_stops=False,
                                     lower=True,
                                     split=True,
                                     min_len=0)
        self.docs = list(self.corpus)

    def test_lazy(self):
        corpus = SnapshotCorpus(repo=self.repo,
                                remove_stops=False,
                                lower=True,
                                split=True,
                                min_len=0,
                                lazy_dict=True)

        self.assertEqual(len(corpus.id2word), 0)

        # if lazy, iterating over the corpus will now build the dict
        docs = list(corpus)

        self.assertGreater(len(corpus.id2word), 0)


    def test_length(self):
        self.assertEqual(len(self.corpus), 10)
        self.assertEqual(len(self.docs), 10)

        l = len(self.corpus)
        for _ in self.corpus:
            self.assertEqual(l, len(self.corpus))

    def test_get_texts(self):
        documents = [
                [u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                [u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                [u'the', u'eps', u'user', u'interface', u'management', u'system'],
                [u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],
                [u'relation', u'of', u'user', u'perceived', u'response', u'time', u'to', u'error', u'measurement'],
                [u'the', u'generation', u'of', u'random', u'binary', u'unordered', u'trees'],
                [u'the', u'intersection', u'graph', u'of', u'paths', u'in', u'trees'],
                [u'graph', u'minors', u'iv', u'widths', u'of', u'trees', u'and', u'well', u'quasi', u'ordering'],
                [u'graph', u'minors', u'a', u'survey'],
                [u'graph', u'minors', u'a', u'survey'],
                [u'graph', u'minors', u'a', u'survey'],
                ]

        for doc in self.corpus.get_texts():
            doc = list(doc) # generators, woo?
            self.assertIn(doc, documents)

    def test_metadata_get_texts(self):
        self.corpus.metadata = True

        documents = [
                ([u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                    ('a/0.txt', u'test_git')),
                ([u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                    ('a/1.txt', u'test_git')),
                ([u'the', u'eps', u'user', u'interface', u'management', u'system'],
                    ('b/2.txt', u'test_git')),
                ([u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],
                    ('b/3.txt', u'test_git')),
                ([u'relation', u'of', u'user', u'perceived', u'response', u'time', u'to', u'error', u'measurement'],
                    ('c/4.txt', u'test_git')),
                ([u'the', u'generation', u'of', u'random', u'binary', u'unordered', u'trees'],
                    ('c/e/5.txt', u'test_git')),
                ([u'the', u'intersection', u'graph', u'of', u'paths', u'in', u'trees'],
                    ('c/f/6.txt', u'test_git')),
                ([u'graph', u'minors', u'iv', u'widths', u'of', u'trees', u'and', u'well', u'quasi', u'ordering'],
                    ('7.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),
                ]

        for docmeta in self.corpus.get_texts():
            doc, meta = docmeta
            doc = list(doc) # generators, woo?
            docmeta = doc, meta # get a non (generator, metadata) pair
            self.assertIn(docmeta, documents)

    def test_docs(self):
        documents = [
                [
                    (u'human', 1),
                    (u'machine', 1),
                    (u'interface', 1),
                    (u'for', 1),
                    (u'lab', 1),
                    (u'abc', 1),
                    (u'computer', 1),
                    (u'applications', 1),
                    ],

                [
                    (u'a', 1),
                    (u'survey', 1),
                    (u'of', 2),
                    (u'user', 1),
                    (u'opinion', 1),
                    (u'computer', 1),
                    (u'system', 1),
                    (u'response', 1),
                    (u'time', 1),
                    ],

                [
                    (u'the', 1),
                    (u'eps', 1),
                    (u'user', 1),
                    (u'interface', 1),
                    (u'management', 1),
                    (u'system', 1),
                    ],

                [
                    (u'system', 2),
                    (u'and', 1),
                    (u'human', 1),
                    (u'engineering', 1),
                    (u'testing', 1),
                    (u'of', 1),
                    (u'eps', 1),
                    ],

                [
                    (u'relation', 1),
                    (u'of', 1),
                    (u'user', 1),
                    (u'perceived', 1),
                    (u'response', 1),
                    (u'time', 1),
                    (u'to', 1),
                    (u'error', 1),
                    (u'measurement', 1),
                    ],

                [
                    (u'the', 1),
                    (u'generation', 1),
                    (u'of', 1),
                    (u'random', 1),
                    (u'binary', 1),
                    (u'unordered', 1),
                    (u'trees', 1),
                    ],

                [
                    (u'the', 1),
                    (u'intersection', 1),
                    (u'graph', 1),
                    (u'of', 1),
                    (u'paths', 1),
                    (u'in', 1),
                    (u'trees', 1),
                    ],

                [
                    (u'graph', 1),
                    (u'minors', 1),
                    (u'iv', 1),
                    (u'widths', 1),
                    (u'of', 1),
                    (u'trees', 1),
                    (u'and', 1),
                    (u'well', 1),
                    (u'quasi', 1),
                    (u'ordering', 1),
                    ],

                [
                    (u'graph', 1),
                    (u'minors', 1),
                    (u'a', 1),
                    (u'survey', 1),
                    ],
                ]

        documents = [set(x) for x in documents]

        for doc in self.corpus:
            self.assertGreater(len(doc), 0)

            # convert the document to text freq since we don't know the
            # term ids ahead of time for testing.
            textdoc = set((unicode(self.corpus.id2word[x[0]]), x[1]) for x in doc)
            self.assertIn(textdoc, documents)

class TestReleaseCorpus(TestSnapshotCorpus):
    def setUp(self):
        super(TestReleaseCorpus, self).setUp()
        project = self.Project(ref=None,
                               level='file',
                               src_path=self.basepath)
        self.corpus = ReleaseCorpus(project=project,
                                    remove_stops=False,
                                    lower=True,
                                    split=True,
                                    min_len=0,
                                    label='test_git')
        self.docs = list(self.corpus)


class TestSnapshotCorpusAtRef(TestGitCorpus):
    def setUp(self):
        super(TestSnapshotCorpusAtRef, self).setUp()
        p1 = self.Project(ref=u'f33a0fb070a34fc1b9105453b3ffb4edc49131d9',
                          level='file',
                          src_path=self.basepath)
        self.corpus = SnapshotCorpus(repo=self.repo,
                                     project=p1,
                                     remove_stops=False,
                                     lower=True,
                                     split=True,
                                     min_len=0)
        self.docs = list(self.corpus)


    def test_lazy(self):
        corpus = SnapshotCorpus(repo=self.repo,
                                remove_stops=False,
                                lower=True,
                                split=True,
                                min_len=0,
                                lazy_dict=True)

        self.assertEqual(len(corpus.id2word), 0)

        # if lazy, iterating over the corpus will now build the dict
        docs = list(corpus)

        self.assertGreater(len(corpus.id2word), 0)

    def test_length(self):
        self.assertEqual(len(self.corpus), 7)
        self.assertEqual(len(self.docs), 7)

        l = len(self.corpus)
        for _ in self.corpus:
            self.assertEqual(l, len(self.corpus))

    def test_get_texts(self):
        documents = [
                [u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                [u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                [u'the', u'eps', u'user', u'interface', u'management', u'system'],
                [u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],
                [u'graph', u'minors', u'a', u'survey'],
                [u'graph', u'minors', u'a', u'survey'],
                [u'graph', u'minors', u'a', u'survey'],
                ]

        for doc in self.corpus.get_texts():
            doc = list(doc) # generators, woo?
            self.assertIn(doc, documents)

    def test_metadata_get_texts(self):
        self.corpus.metadata = True

        documents = [
                ([u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                    ('a/0.txt', u'test_git')),
                ([u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                    ('a/1.txt', u'test_git')),
                ([u'the', u'eps', u'user', u'interface', u'management', u'system'],
                    ('b/2.txt', u'test_git')),
                ([u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],
                    ('b/3.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),
                ]

        for docmeta in self.corpus.get_texts():
            doc, meta = docmeta
            doc = list(doc) # generators, woo?
            docmeta = doc, meta # get a non (generator, metadata) pair
            self.assertIn(docmeta, documents)

    def test_docs(self):
        documents = [
                [
                    (u'human', 1),
                    (u'machine', 1),
                    (u'interface', 1),
                    (u'for', 1),
                    (u'lab', 1),
                    (u'abc', 1),
                    (u'computer', 1),
                    (u'applications', 1),
                    ],

                [
                    (u'a', 1),
                    (u'survey', 1),
                    (u'of', 2),
                    (u'user', 1),
                    (u'opinion', 1),
                    (u'computer', 1),
                    (u'system', 1),
                    (u'response', 1),
                    (u'time', 1),
                    ],

                [
                    (u'the', 1),
                    (u'eps', 1),
                    (u'user', 1),
                    (u'interface', 1),
                    (u'management', 1),
                    (u'system', 1),
                    ],

                [
                    (u'system', 2),
                    (u'and', 1),
                    (u'human', 1),
                    (u'engineering', 1),
                    (u'testing', 1),
                    (u'of', 1),
                    (u'eps', 1),
                    ],

                [
                    (u'graph', 1),
                    (u'minors', 1),
                    (u'a', 1),
                    (u'survey', 1),
                    ],
                ]

        documents = [set(x) for x in documents]

        for doc in self.corpus:
            self.assertGreater(len(doc), 0)

            # convert the document to text freq since we don't know the
            # term ids ahead of time for testing.
            textdoc = set((unicode(self.corpus.id2word[x[0]]), x[1]) for x in doc)
            self.assertIn(textdoc, documents)


class TestChangesetCorpus(TestGitCorpus):
    def setUp(self):
        super(TestChangesetCorpus, self).setUp()
        self.corpus = ChangesetCorpus(repo=self.repo,
                                      remove_stops=False,
                                      lower=True,
                                      split=True,
                                      min_len=0)
        self.docs = list(self.corpus)


    def test_length(self):
        self.assertEqual(len(self.corpus), 5)
        self.assertEqual(len(self.docs), 5)

        l = len(self.corpus)
        for _ in self.corpus:
            self.assertEqual(l, len(self.corpus))

    def test_lazy(self):
        corpus = SnapshotCorpus(repo=self.repo,
                                remove_stops=False,
                                lower=True,
                                split=True,
                                min_len=0,
                                lazy_dict=True)

        self.assertEqual(len(corpus.id2word), 0)

        # if lazy, iterating over the corpus will now build the dict
        docs = list(corpus)

        self.assertGreater(len(corpus.id2word), 0)


    def test_changeset_get_texts(self):
        documents = [
                # systems
                [u'graph', u'minors', u'a', u'survey'] +
                [u'graph', u'minors', u'a', u'survey'] +
                [u'graph', u'minors', u'a', u'survey'],

                # a/
                [u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'] +
                [u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],

                # b/
                [u'the', u'eps', u'user', u'interface', u'management', u'system'] +
                [u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],

                # c/
                # TODO apparently file c/4.txt is fubar in the test repo
                # [u'relation', u'of', u'user', u'perceived', u'response', u'time', u'to', u'error', u'measurement'] +
                [u'the', u'generation', u'of', u'random', u'binary', u'unordered', u'trees'] +
                [u'the', u'intersection', u'graph', u'of', u'paths', u'in', u'trees'],

                # 7
                [u'graph', u'minors', u'iv', u'widths', u'of', u'trees', u'and', u'well', u'quasi', u'ordering'],

                ]

        documents = [list(sorted(x)) for x in documents]

        for i, doc in enumerate(self.corpus.get_texts()):
            doc = list(sorted(doc)) # generators, woo?
            self.assertEqual(doc, documents[i])

    def test_changeset_metadata_get_texts(self):
        self.corpus.metadata = True

        documents = [
                # systems
                (
                    [u'graph', u'minors', u'a', u'survey'] +
                    [u'graph', u'minors', u'a', u'survey'] +
                    [u'graph', u'minors', u'a', u'survey'],
                    (u'2aeb2e7c78259833e1218b69f99dab3acd00970c', u'test_git')),

                # a/
                (
                    [u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'] +
                    [u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                    (u'3587d37e7d476ddc7b673c41762dc89c8ca63a6a', u'test_git')),

                # b/
                (
                    [u'the', u'eps', u'user', u'interface', u'management', u'system'] +
                    [u'system', u'and', u'human', u'system', u'engineering', u'testing', u'of', u'eps'],
                    (u'f33a0fb070a34fc1b9105453b3ffb4edc49131d9', u'test_git')),

                # c/
                # TODO apparently file c/4.txt is fubar in the test repo
                (
                    #[u'relation', u'of', u'user', u'perceived', u'response', u'time', u'to', u'error', u'measurement'] +
                    [u'the', u'generation', u'of', u'random', u'binary', u'unordered', u'trees'] +
                    [u'the', u'intersection', u'graph', u'of', u'paths', u'in', u'trees'],
                    (u'899268bdd33aec225f6264a734dac2081f78ab54', u'test_git')),

                # 7
                (
                    [u'graph', u'minors', u'iv', u'widths', u'of', u'trees', u'and', u'well', u'quasi', u'ordering'],
                    (u'f870a217765a268fe5c5315d58ef671050d17fb9', u'test_git')),

                ]

        documents = [(list(sorted(x[0])), x[1]) for x in documents]

        for docmeta in self.corpus.get_texts():
            doc, meta = docmeta
            doc = list(sorted(doc)) # generators, woo?
            docmeta = doc, meta # get a non (generator, metadata) pair
            self.assertIn(docmeta, documents)


    def test_changeset_docs(self):
        documents = [
                [
                    (u'and', 1),
                    (u'graph', 1),
                    (u'iv', 1),
                    (u'minors', 1),
                    (u'of', 1),
                    (u'ordering', 1),
                    (u'quasi', 1),
                    (u'trees', 1),
                    (u'well', 1),
                    (u'widths', 1),
                    ],

                [
                    (u'binary', 1),
                    # (u'error', 1),
                    (u'generation', 1),
                    (u'graph', 1),
                    (u'in', 1),
                    (u'intersection', 1),
                    # (u'measurement', 1),
                    # (u'of', 3),
                    (u'of', 2),

                    (u'paths', 1),
                    # (u'perceived', 1),
                    (u'random', 1),
                    # (u'relation', 1),
                    # (u'response', 1),
                    (u'the', 2),
                    # (u'time', 1),
                    # (u'to', 1),
                    (u'trees', 2),
                    (u'unordered', 1),
                    # (u'user', 1),
                    ],

                # [u'relation', u'of', u'user', u'perceived', u'response', u'time', u'to', u'error', u'measurement'] +

                [
                    (u'and', 1),
                    (u'engineering', 1),
                    (u'eps', 2),
                    (u'human', 1),
                    (u'interface', 1),
                    (u'management', 1),
                    (u'of', 1),
                    (u'system', 3),
                    (u'testing', 1),
                    (u'the', 1),
                    (u'user', 1),
                    ],

                [
                    (u'a', 1),
                    (u'abc', 1),
                    (u'applications', 1),
                    (u'computer', 2),
                    (u'for', 1),
                    (u'human', 1),
                    (u'interface', 1),
                    (u'lab', 1),
                    (u'machine', 1),
                    (u'of', 2),
                    (u'opinion', 1),
                    (u'response', 1),
                    (u'survey', 1),
                    (u'system', 1),
                    (u'time', 1),
                    (u'user', 1),
                    ],

                [
                    (u'a', 3),
                    (u'graph', 3),
                    (u'minors', 3),
                    (u'survey', 3),
                    ],

                ]

        documents = [set(x) for x in documents]

        for doc in self.corpus:
            self.assertGreater(len(doc), 0)

            # convert the document to text freq since we don't know the
            # term ids ahead of time for testing.
            textdoc = set((unicode(self.corpus.id2word[x[0]]), x[1]) for x in doc)
            self.assertIn(textdoc, documents)


class TestCorpusCombiner(TestGitCorpus):
    def setUp(self):
        super(TestCorpusCombiner, self).setUp()
        # 3 documents
        p1 = self.Project(ref=u'2aeb2e7c78259833e1218b69f99dab3acd00970c',
                          level='file',
                          src_path=self.basepath)
        self.corpus1 = SnapshotCorpus(repo=self.repo,
                                      project=p1,
                                      remove_stops=False,
                                      lower=True,
                                      split=True,
                                      min_len=0)
        self.docs1 = list(self.corpus1)

        # 3 old documents + 2 new documents
        p2 = self.Project(ref=u'3587d37e7d476ddc7b673c41762dc89c8ca63a6a',
                          level='file',
                          src_path=self.basepath)
        self.corpus2 = SnapshotCorpus(repo=self.repo,
                                      project=p2,
                                      remove_stops=False,
                                      lower=True,
                                      split=True,
                                      min_len=0)
        self.docs2 = list(self.corpus2)


        self.corpus = CorpusCombiner([self.corpus1, self.corpus2])
        self.docs = list(self.corpus)

    def test_length(self):
        self.assertEqual(len(self.corpus), 8)
        self.assertEqual(len(self.docs), 8)

        l = len(self.corpus)
        for _ in self.corpus:
            self.assertEqual(l, len(self.corpus))

    def test_metadata_docs(self):
        documents = [
                # corpus1
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),

                # corpus2
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),
                ([u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                    ('a/0.txt', u'test_git')),
                ([u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                    ('a/1.txt', u'test_git')),
                ]

        documents = [(set(x),y) for x,y in documents]

        self.corpus.metadata = True
        vals = [self.corpus.metadata,
                self.corpus._metadata,
                self.corpus.corpora[0].metadata,
                self.corpus.corpora[1].metadata]
        self.assertTrue(all(vals))

        for docmeta in self.corpus:
            doc, meta = docmeta
            self.assertGreater(len(doc), 0)

            # convert the document to text freq since we don't know the
            # term ids ahead of time for testing.
            textdoc = set(unicode(self.corpus.id2word[x[0]]) for x in doc)
            docmeta = textdoc, meta
            self.assertIn(docmeta, documents)

    def test_mallet_corpus(self):
        with open(self.tempfname, 'w') as f:
            f.write('abc en fred flintstone\n')
            f.write('efg en barney rubble\n')
        corpus3 = MalletCorpus(self.tempfname)
        self.assertEqual(len(corpus3), 2)
        self.corpus.add(corpus3)
        self.assertEqual(len(self.corpus), 10)

        documents = [
                # corpus1
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),

                # corpus2
                ([u'graph', u'minors', u'a', u'survey'],
                    ('dos.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('mac.txt', u'test_git')),
                ([u'graph', u'minors', u'a', u'survey'],
                    ('unix.txt', u'test_git')),
                ([u'human', u'machine', u'interface', u'for', u'lab', u'abc', u'computer', u'applications'],
                    ('a/0.txt', u'test_git')),
                ([u'a', u'survey', u'of', u'user', u'opinion', u'of', u'computer', u'system', u'response', u'time'],
                    ('a/1.txt', u'test_git')),

                # mallet
                ([u'fred', u'flintstone'],
                    ('abc', u'en')),
                ([u'barney', u'rubble'],
                    ('efg', u'en')),
                ]

        documents = [(set(x),y) for x,y in documents]

        self.corpus.metadata = True
        vals = [self.corpus.metadata,
                self.corpus._metadata,
                self.corpus.corpora[0].metadata,
                self.corpus.corpora[1].metadata,
                self.corpus.corpora[2].metadata]
        self.assertTrue(all(vals))

        for docmeta in self.corpus:
            doc, meta = docmeta
            self.assertGreater(len(doc), 0)

            # convert the document to text freq since we don't know the
            # term ids ahead of time for testing.
            textdoc = set(unicode(self.corpus.id2word[x[0]]) for x in doc)
            docmeta = textdoc, meta
            self.assertIn(docmeta, documents)
