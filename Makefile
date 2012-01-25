all: develop
	
develop:
	python setup.py develop

install:
	python setup.py install

test: test-local test-bo
	
test-local:
	nosetests -a '!slow' boot_agents
	
test-bo:
	BO_TEST_CONFIG=$(PWD)/src/boot_agents/configs/for_testing nosetests bootstrapping_olympics
	