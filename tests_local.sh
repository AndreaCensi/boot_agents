#!/bin/bash
set -e
set -x

nosetests -w src -v -a '!slow'