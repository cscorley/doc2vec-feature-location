all: test 

web: always_make
	mkdir -p web
	git archive -o web/release.tar.gz --prefix="cfl/" HEAD
	make paper
	cp paper/paper.pdf web/
	pandoc -s -o web/index.html README.md --css="/css/pure-min.css" --css="/css/github.css" --css="/css/styles.css"

clean-web:
	rm -rf web/


paper: always_make
	cd paper/ && make

always_make:
	true

test:
	nosetests tests/ || true
	find src tests -name '*.pyc' -exec rm {} \;

clean: clean-web clean-corpora clean-models
	find src tests -name '*.pyc' -exec rm {} \;

clean-corpora: clean-taser
	find data -name '*.mallet.gz.index' -exec rm {} \;
	find data -name '*.mallet.gz' -exec rm {} \;
	find data -name '*.dict.gz' -exec rm {} \;

clean-models: clean-lda clean-lsi

clean-taser:
	rm -rf /tmp/taser_*

clean-lda:
	find data -name 'LDA*' -exec rm {} \;
	find data -name '*.lda*' -exec rm {} \;

clean-lsi:
	find data -name 'LSI*' -exec rm {} \;
	find data -name '*.lsi*' -exec rm {} \;


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

