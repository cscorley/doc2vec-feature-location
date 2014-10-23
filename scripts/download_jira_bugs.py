from __future__ import print_function

from lxml import etree
from StringIO import StringIO
import codecs

from src.utils import mkdir
from src.preprocessing import to_unicode
import sys
import requests


# 3 python-code lines commented out to download the hibernate files
url_base = 'https://issues.apache.org/jira/si/jira.issueviews:issue-xml/%s/%s.xml'
#url_base = 'https://hibernate.atlassian.net/si/jira.issueviews:issue-xml/%s/%s.xml'
projects = [('bookkeeper', 'v4.1.0'),
            ('derby', 'v10.7.1.1'),
            ('derby', 'v10.9.1.0'),
            ('lucene', 'v4.0'),
            ('mahout', 'v0.8'),
            ('openjpa', 'v2.0.1'),
            ('openjpa', 'v2.2.0'),
            ('pig', 'v0.8.0'),
            ('pig', 'v0.11.1'),
            ('solr', 'v4.4.0'),
            ('tika', 'v1.3'),
            ('zookeeper', 'v3.4.5'),
            ]

#projects = [('hibernate', 'v3.5.0b2')]
for project, version in projects:
    path = '/'.join(['data', project, version])
    print(path)
    mkdir(path + '/queries')

    with open(path + '/ids.txt') as f:
        bugs = [x.strip() for x in f]

    p = etree.XMLParser()
    hp = etree.HTMLParser()

    for bugid in bugs:
        print("Fetching bugid", bugid)
        fname = project.upper() + '-' + bugid
#        fname = 'HHH-' + bugid
        r = requests.get(url_base % (fname, fname))
        try:
            tree = etree.parse(StringIO(r.text), p)
        except etree.XMLSyntaxError:
            print("ERROR", bugid, project, version)
            continue
        root = tree.getroot()
        html = root.find('channel').find('item').find('description').text
        summary = root.find('channel').find('item').find('summary').text
        summary = to_unicode(summary)

        htree = etree.parse(StringIO(html), hp)
        desc = ''.join(htree.getroot().itertext())
        desc = to_unicode(desc)


        with codecs.open(path + '/queries/ShortDescription%s.txt' % bugid, 'w', 'utf-8') as f:
            f.write(summary)

        with codecs.open(path + '/queries/LongDescription%s.txt' % bugid, 'w', 'utf-8') as f:
            f.write(desc)
