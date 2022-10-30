#!/bin/bash

echo -n " Upisi godinu za koju te zanima koliko puta se dogodio petak 13.: "
read godina ;
echo -n "Upisi godinu za do koje te zanima koliko puta ce se desiti petak 13."
read opseg

suma=0
for i in $(seq $godina $opseg);
do
  petkovi=$( ncal $godina | grep pe | grep -c 13);
  suma=$(($suma+$petkovi))
done

echo "od $godina.godine do $opseg.godine ima $suma petka trinaestih"


