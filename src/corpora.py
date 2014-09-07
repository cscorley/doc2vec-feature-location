#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

"""
Code for generating the corpora.
"""

from StringIO import StringIO
import re
import subprocess

import gensim
import dulwich
import dulwich.index
import dulwich.repo
import dulwich.patch

import preprocessing

import logging
logger = logging.getLogger('cfl.corpora')


class GitCorpus(gensim.interfaces.CorpusABC):

    """
    Helper class to simplify the pipeline of getting bag-of-words vectors (=
    a gensim corpus) from plain text.

    This is an abstract base class: override the `get_texts()` method to match
    your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object
    will be automatically initialized with a dictionary in `self.dictionary` and
    will support the `iter` corpus method. You must only provide a correct
    `get_texts` implementation.
    """

    def __init__(self, repo=None, ref='HEAD', remove_stops=True,
                 split=True, lower=True, min_len=3, max_len=40,
                 lazy_dict=False):

        logger.info('Creating %s corpus out of source files for commit %s' % (
            self.__class__.__name__, ref))

        self.repo = repo
        self.remove_stops = remove_stops
        self.split = split
        self.lower = lower
        self.min_len = min_len
        self.max_len = max_len
        self.lazy_dict = lazy_dict

        self.id2word = gensim.corpora.Dictionary()
        self.metadata = False

        # ensure ref is a str otherwise dulwich cries
        if isinstance(ref, unicode):
            self.ref = ref.encode('utf-8')
        else:
            self.ref = ref

        self.ref_tree = None

        if repo is not None:
            # find which file tree is for the commit we care about
            self.ref_tree = self.repo[self.ref].tree

            if not lazy_dict:
                # build the dict (not lazy)
                self.id2word.add_documents(self.get_texts())

        super(GitCorpus, self).__init__()

    def preprocess(self, document, info=[]):
        document = preprocessing.to_unicode(document, info)
        words = preprocessing.tokenize(document)

        if self.split:
            words = preprocessing.split(words)

        if self.lower:
            words = (word.lower() for word in words)

        if self.remove_stops:
            words = preprocessing.remove_stops(words, preprocessing.FOX_STOPS)
            words = preprocessing.remove_stops(words, preprocessing.JAVA_RESERVED)

        def include(word):
            return len(word) >= self.min_len and len(word) <= self.max_len
        words = (word for word in words if include(word))
        return words

    def __iter__(self):
        """
        The function that defines a corpus.

        Iterating over the corpus must yield sparse vectors, one for each
        document.
        """
        for text in self.get_texts():
            if self.metadata:
                meta = text[1]
                text = text[0]
                yield (self.id2word.doc2bow(text, allow_update=self.lazy_dict),
                       meta)
            else:
                yield self.id2word.doc2bow(text, allow_update=self.lazy_dict)

    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into
        `Dictionary.doc2bow`.

        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no
        further preprocessing of the words coming out of this function.
        """
        raise NotImplementedError

    def __len__(self):
        return self.length  # will throw if corpus not initialized


class SnapshotCorpus(GitCorpus):

    def get_texts(self):
        length = 0

        for entry in self.repo.object_store.iter_tree_contents(self.ref_tree):
            fname = entry.path
            document = self.repo.object_store.get_raw(entry.sha)[1]
            if dulwich.patch.is_binary(document):
                continue

            words = self.preprocess(document, [fname, self.ref])
            length += 1

            if self.metadata:
                yield words, (fname, u'en')
            else:
                yield words

        self.length = length  # only reset after iteration is done.


class TaserSnapshotCorpus(GitCorpus):

    def __init__(self, repo=None, ref='HEAD', remove_stops=True,
                 split=True, lower=True, min_len=3, max_len=40,
                 taser_jar='lib/taser.jar'):
        # force lazy_dict since we have to run taser to build the corpus first
        super(TaserSnapshotCorpus, self).__init__(repo, ref, remove_stops,
                                                  split, lower, min_len,
                                                  max_len, lazy_dict=True)
        self.taser_jar = taser_jar

        # checkout the version we want
        dulwich.index.build_index_from_tree(self.repo.path,
                                            self.repo.index_path(),
                                            self.repo.object_store,
                                            self.ref_tree)
        self.corpus_generated = False

    def run_taser(self):
        self.corpus_generated = False

        cmds = list()
        cmds.append(['java', '-jar', self.taser_jar, 'ex', self.repo.path, '-o', 'taser_output'])
        cmds.append(['java', '-jar', self.taser_jar, 'rw', 'taser_output', '-o', 'taser_output'])
        cmds.append(['java', '-jar', self.taser_jar, 'bc', 'taser_output', '-o', 'taser_output'])
        # do not need to do pp since we will preproccess ourselves
        #cmds.append(['java', '-jar', self.taser_jar, 'pp', 'taser_output', '-o', 'taser_output'])
        cmds.append(['java', '-jar', self.taser_jar, 'fc', 'taser_output', '-o', 'taser_output'])

        for cmd in cmds:
            retval = subprocess.call(cmd)
            if retval:
                raise Exception("Failed cmd: " + str(cmd))

        self.corpus_generated = True

    def get_texts(self):
        if not self.corpus_generated:
            self.run_taser()

        length = 0

        with open('taser_output/unknown-0.0.ser') as f:
            for line in f:
                doc_name, document = line.split(' ', 1)
                words = self.preprocess(document, [doc_name, self.ref])
                length += 1

                if self.metadata:
                    yield words, (doc_name, u'en')
                else:
                    yield words

        self.length = length  # only reset after iteration is done.


class ChangesetCorpus(GitCorpus):

    def _get_diff(self, changeset):
        """ Return a text representing a `git diff` for the files in the
        changeset.

        """
        patch_file = StringIO()
        dulwich.patch.write_object_diff(patch_file,
                                        self.repo.object_store,
                                        changeset.old, changeset.new)
        return patch_file.getvalue()

    def _walk_changes(self, reverse=False):
        """ Returns one file change at a time, not the entire diff.

        """

        for walk_entry in self.repo.get_walker(reverse=reverse):
            commit = walk_entry.commit

            # initial revision, has no parent
            if len(commit.parents) == 0:
                for changes in dulwich.diff_tree.tree_changes(
                        self.repo.object_store, None, commit.tree
                ):
                    diff = self._get_diff(changes)
                    yield commit.id, None, diff

            for parent in commit.parents:
                # do I need to know the parent id?

                for changes in dulwich.diff_tree.tree_changes(
                    self.repo.object_store, self.repo[parent].tree, commit.tree
                ):
                    diff = self._get_diff(changes)
                    yield commit.id, parent, diff

    def get_texts(self):
        length = 0
        unified = re.compile(r'^[+ -].*')
        current = None
        low = list()  # collecting the list of words

        for commit, parent, diff in self._walk_changes():
            # write out once all diff lines for commit have been collected
            # this is over all parents and all files of the commit
            if current is None:
                # set current for the first commit, clear low
                current = commit
                low = list()
            elif current != commit:
                # new commit seen, yield the collected low
                if self.metadata:
                    yield low, (current, u'en')
                else:
                    yield low

                length += 1
                current = commit
                low = list()

            # to process out whitespace only changes, the rest of this
            # loop will need to be structured differently. possibly need
            # to actually parse the diff to gain structure knowledge
            # (ie, line numbers of the changes).

            diff_lines = filter(lambda x: unified.match(x),
                                diff.splitlines())
            if len(diff_lines) < 2:
                continue  # useful for not worrying with binary files

            # sanity?
            assert diff_lines[0].startswith('--- '), diff_lines[0]
            assert diff_lines[1].startswith('+++ '), diff_lines[1]
            # parent_fn = diff_lines[0][4:]
            # commit_fn = diff_lines[1][4:]

            lines = diff_lines[2:]  # chop off file names hashtag rebel
            lines = [line[1:] for line in lines]  # remove unified markers
            document = ' '.join(lines)

            # call the tokenizer
            words = self.preprocess(document,
                                    [commit, str(parent), diff_lines[0]])
            low.extend(words)

        length += 1
        if self.metadata:
            # have reached the end, yield whatever was collected last
            yield low, (current, u'en')
        else:
            yield low

        self.length = length  # only reset after iteration is done.


class CommitLogCorpus(GitCorpus):

    def get_texts(self):
        length = 0

        for walk_entry in self.repo.get_walker():
            commit = walk_entry.commit
            words = self.preprocess(commit.message, [commit.id])

            length += 1
            if self.metadata:
                # have reached the end, yield whatever was collected last
                yield words, (commit.id, u'en')
            else:
                yield words

        self.length = length  # only reset after iteration is done.
