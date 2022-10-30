#!/bin/bash

# Ispis PID -a skripte
# Inicijalizacija varijabli i postavljanje signal handlera

touch /tmp/processPID
echo $$ > /tmp/processPID

t1=0
t2=0
t3=0

trap 't1=$((t1+1))' 1
trap 't2=$((t2+1))' 3
trap 't3=$((t3+1))' 4

while true
do
	echo t1: $t1;
    echo t2: $t2;
    echo t3: $t3;
	echo --------
sleep 1
done
