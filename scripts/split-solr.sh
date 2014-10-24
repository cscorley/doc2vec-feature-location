#!/bin/bash

# similar commands used to split lucene from same repo
git clone git://github.com/apache/lucene-solr solr-only
cd solr-only
git filter-branch --prune-empty --subdirectory-filter solr
git fsck --full
git gc
git remote add mirror git@github.com:cscorley/solr-only-mirror.git
git push --mirror mirror
