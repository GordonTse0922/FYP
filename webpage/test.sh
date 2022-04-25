#!/bin/bash

for i in `find . -name '*.py'`
do
    python $i
done
