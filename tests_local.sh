#!/bin/bash
set -e
set -x

nosetests -w src -a '!slow'