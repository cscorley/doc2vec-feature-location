all: test

test:
	nosetests tests/ || true
	find src tests -name '*.pyc' -exec rm {} \;

clean: clean-corpora clean-models
	find src tests -name '*.pyc' -exec rm {} \;

clean-corproa:
	find data -name '*.mallet.gz' -exec rm {} \;
	find data -name '*.dict.gz' -exec rm {} \;

clean-models:
	find data -name '*.lda*' -exec rm {} \;


install: requirements
	pip install --editable .

init:
	virtualenv --python=python2 env
	. env/bin/activate && pip install numpy
	. env/bin/activate && pip install -r requirements.txt
	. env/bin/activate && pip install --editable .

requirements:
	pip install numpy
	pip install -r requirements.txt

