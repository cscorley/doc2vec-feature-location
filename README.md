Feature location with changeset topics
===============

Basic idea: compare LDA models built with two different sources of
documents. The first being a traditional source code snapshot, and the
second being variations on a changeset snapshot.

### Dependencies

You need to already have `antlr4` on your system in order to build the
Java grammar. How you accomplish this is up to you.

Build the Java grammar with:

    antlr4 -Dlanguage=Python2 grammars/Java.g4 -o src/

Your compiled grammar should end up in `src/grammars/`

### Installing

Install everything using make:

    $ make install

Or, if you use virtualenv, you can `make init` instead.

Now, you should be able to run commands:

    $ cfl --help

    Usage: cfl [OPTIONS] COMMAND [ARGS]...

      Changeset Feature Location

    Options:
      --verbose
      --help     Show this message and exit.

    Commands:
      corpora     Builds the basic corpora for a project
      model       Builds a model for the corpora
      evaluate    Evalutates the models
      run_all     Runs corpora, preprocess, model, and evaluate...
