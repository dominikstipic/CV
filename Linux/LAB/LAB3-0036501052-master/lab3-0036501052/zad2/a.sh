#!/bin/bash


pid=$(cat /tmp/processPID)
echo "pid je " $pid

while true
do 
	sleep 1
	sigval=$((1+RANDOM %3))
	case $sigval in 
		1)
			kill -1 $pid 		
			;;
		2) 
			kill -3 $pid
			;;
		3)
			kill -4 $pid
			;; 
	esac
done
