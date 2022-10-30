#!/bin/bash

cd /tmp/OKOSL/downloads;

for i in $(ls)
do 
x=$(echo $i | egrep "_");

if  [ $x ]
    then
       ime_predmeta=$(echo $i | cut -d"_" -f1);
       ime_dat=$(echo $i | cut --complement -d"_" -f 1);

	   if [ ! -d /tmp/OKOSL/downloads/$ime_predmeta ]
		then 	
			mkdir $ime_predmeta;
		fi
       
       mv $i $ime_predmeta;
fi
done

mkdir razonoda
mv *.* razonoda


for i in razonoda/*
do 
x=$(echo $i | cut -d. -f 2)
echo $x

if [ $x = "pdf" ] || [ $x = "epub" ]
then
	if [ ! -d knjige ]
	then 
		mkdir knjige
	fi
	mv $i knjige
fi

if [ $x = "jpg" ] || [ $x = "jpeg" ]
then
	if [ ! -d slike ]
	then 
		mkdir slike
	fi
	mv $i slike
fi

if [ $x = "mp3" ] 
then
	if [ ! -d muzika ]
	then 
		mkdir muzika
	fi
	mv $i muzika
fi
done
