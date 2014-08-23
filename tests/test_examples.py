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

from nose.tools import *
import unittest

import src

class TestExample(unittest.TestCase):
    def setUp(self):
        self.is_setup = True

    def test_truth(self):
        assert self.is_setup

    @raises(AssertionError)
    def test_passes_by_failing(self):
        assert not self.is_setup
