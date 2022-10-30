#!/bin/bash

# Ctrl + Z -> proces je zaustavljen
# bg [jobid] -> proces ide u pozadinu
# fg [jobid] -> procse id u frground
# ./lab3.sh & -> proces se pokrece u pozadini


pid=$(pgrep lab3.sh);
echo "pid: "$pid

echo "old nice"
ps -p $pid -o nice
renice 15 $pid
ps -p $pid -o nice

# prilikom gašenja terminala procesima u terminalu se salje SIGHUP

#nohup bash lab3.sh &
# pokreće proces u pozadini i zapisuje izlaz u nohup.out datoteku

