#! /bin/bash
a=0
while true; do
  printf "%d\r" $a
  a=$[$a+1]
  sleep 1
done
