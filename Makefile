all:
	nosetests tests/ || true
	find src tests -name '*.pyc' -exec rm {} \;

install: submodules requirements
	pip install --editable .

submodules:
	git submodule init
	git submodule update

update:
	git submodule foreach git pull origin master

init: submodules
	virtualenv --python=python2 env
	. env/bin/activate && pip install numpy
	. env/bin/activate && pip install -r requirements.txt
	. env/bin/activate && pip install --editable .

requirements:
	pip install numpy
	pip install -r requirements.txt
