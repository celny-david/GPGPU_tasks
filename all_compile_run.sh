#!/bin/bash

# this is an easy script to test compilation and running of all tasks sources in the project

for f in *.cu; do
	echo "===== ===== ====="
	echo "processing : "$f
	nvcc $f
	./a.out
	sleep 0.2
done
