#!/bin/bash

read myInput;

if [ $myInput = 0 ]
then 
	pythonFiles=$(locate -b "*.py");
	for i in $pythonFiles;
	do
		 cat $i | sed -r 's|^[ ]*||' | egrep "^def" | sed -r "s|(def) (.*)([(].*[)]:)|\2|";
	done

elif [ $myInput = 1 ]
then 
    cFiles=$(locate -b "*.c");
	for i in $cFiles;
	do
		cat $i | sed -r 's|^[ ]*||' | egrep "^#" | sed -r "s|(#)(.*)|\2|";
	done

elif [ $myInput = 2 ]
then
	cFiles=$(locate -b "*.c");
	for i in $cFiles;
	do
		x=$(cat $i | egrep -n "#include" | sed -r "s:([0-9]+)(.*):\1:") ;
		x=$(echo $x | sed -r "s:^[ ]*$::") ;
		echo $x;
	done
fi


