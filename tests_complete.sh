#!/bin/bash
set -e
set -x

# Run the boot_olympics tests for all the agents here
BO_TEST_CONFIG=`pwd`/src/boot_agents/configs/for_testing nosetests -w ../bootstrapping_olympics/src $*


# add -a agent=<agent name>
# add -a robot=<id_robot>

