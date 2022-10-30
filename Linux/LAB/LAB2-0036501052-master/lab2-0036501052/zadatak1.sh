#!/bin/bash


echo "broj linija u datoteci /usr/include/stdio.h:"
cat /usr/include/stdio.h | grep "^\s.*" | wc -l

echo "broj linijskih komentara:"
cat /usr/include/stdio.h | egrep "^[#].*" | wc -l

echo "broj blok komentara:"
cat /usr/include/stdio.h | egrep "/\\*.*/*\/" | wc -l





