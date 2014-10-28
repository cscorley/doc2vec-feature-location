"""
This script will go through the commit logs for projects we dont have trace
links for [Moreno et al] and do our best to guess at them.
"""

import dulwich.repo
import re
import csv
from src.main import load_projects, load_repos, load_goldsets
import os.path
from src.utils import clone

projects = load_projects()
for project in projects:
    dest_fn = os.path.join(project.full_path, 'issue2git.csv')
    if os.path.exists(dest_fn):
        continue
    if project.name == 'eclipse':
        continue

    repos = load_repos(project)
    golds = load_goldsets(project)
    ids = set(i for i,g in golds)
    i2g = dict.fromkeys(ids)
    for k in i2g:
        i2g[k] = set()
    for repo in repos:
        #b = re.compile('BOOKKEEPER-([\d]+):')
        #b = re.compile('ZOOKEEPER-([\d]+)')
        b = re.compile('%s-([\d]+)' % project.name.upper())

        for entry in repo.get_walker():
            a = entry.commit
            for issue in b.findall(a.message):
                if issue in i2g:
                    i2g[issue].add(a.id)

    with open(dest_fn, 'w') as f:
        w = csv.writer(f)
        for issue, gits in i2g.items():
            if gits:
                w.writerow([issue] + list(gits))
