Feature location with changeset topics
===============

Basic idea: compare LDA models built with two different sources of
documents. The first being a traditional source code snapshot, and the
second being variations on a changeset snapshot.

### Installing

Install everything using make:

    $ make install

Or, if you use virtualenv, you can `make init` instead.

Now, you should be able to run commands:

    $ flt --help

    Usage: flt [OPTIONS] COMMAND [ARGS]...

      Modeling Changeset Topics

    Options:
      --verbose
      --help     Show this message and exit.

    Commands:
      corpora     Builds the basic corpora for a project
      evaluate    Evalutates the models
      model       Builds a model for the corpora
      preprocess  Runs the preprocessing steps on a corpus
      run_all     Runs corpora, preprocess, model, and evaluate...
