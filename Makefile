package=boot_agents
include pypackage.mk

# XXX: maybe this does not work
test: test-local test-bo
	
test-local:
	nosetests -a '!slow' boot_agents
	
test-bo:
	BO_TEST_CONFIG=$(PWD)/src/boot_agents/configs/for_testing nosetests bootstrapping_olympics
	