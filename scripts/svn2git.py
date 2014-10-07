
from __future__ import print_function

import sys
files = sys.argv[1:-1]
output = sys.argv[-1]

seen = set()

with open(output, 'w') as out:
    for fn in files:
        with open(fn) as f:
            for line in f:
                gitline, svn = line.split('\t')
                svn = svn.strip()
                svn = svn.strip('r')
                _, git, _ = line.split(' ', 2)
                if svn not in seen:
                    seen.add(svn)
                else:
                    print('seen already: ', svn, git)

                print(','.join([svn,git]), file=out)
