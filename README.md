Feature location with changeset topics
===============

[![Build Status](https://magnum.travis-ci.com/cscorley/feature-location.svg?token=CjXwzG3DCkMLgDhpEr4U&branch=master)](https://magnum.travis-ci.com/cscorley/feature-location)


Basic idea: compare LDA models built with two different sources of
documents. The first being a traditional source code snapshot, and the
second being variations on a changeset snapshot.

### Installing

Install everything using make:

    $ make install

Or, if you use virtualenv, you can `make init` instead.

Now, you should be able to run commands:

    $ cfl <project name>

e.g.,

    $ cfl mucommander

This will run the experiment on the given project name. All project
information is in `projects.csv`, and supplementary data is under the
`data` directory:
    - the repository URLs under `data/<project>/repos.txt`
    - extracted release source code `data/<project>/<version>/src`

To run the experiment temporally, use the `--temporal` flag.

    $ cfl --temporal mucommander

See `--help` for additional usage.

    $ cfl --help
    Usage: cfl [OPTIONS] NAME

    Changesets for Feature Location

    Options:
    --verbose
    --debug
    --temporal
    --path TEXT     Set the directory to work within
    --version TEXT  Version of project to run experiment on
    --level TEXT    Granularity level of project to run experiment on
    --help          Show this message and exit.
