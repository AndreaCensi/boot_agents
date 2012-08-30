package=boot_agents
include pypackage.mk

# XXX: maybe this does not work
#test:: test-bo
	
test-local:
	nosetests -a '!slow' boot_agents
	
current_dir=$(dir $(CURDIR)/$(lastword $(MAKEFILE_LIST)))
 
test-bo:
	BO_TEST_CONFIG=$(current_dir)/src/boot_agents/configs/for_testing nosetests bootstrapping_olympics
	