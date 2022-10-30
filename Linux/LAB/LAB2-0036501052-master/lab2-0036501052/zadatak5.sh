#!/bin/bash



imgs=$(locate  PNG*)

for i in $imgs;
do
x=$(echo $i | sed -r "s|(.*)(PNG-)([0-9]{2})([0-9]{2})([0-9]{4})|\1\3_\4_\5.png|");
echo $x
mv $i $x
done
