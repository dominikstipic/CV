#!/bin/bash

mkdir LAB1
cd LAB1
mkdir source 
touch source/empty
cp -r /boot/* /etc/* source
du -h source
ln -s source target
cd target 
pwd
cd ..
cd source 
pwd
cd ..
du -Dh target
touch -r source/empty source/novi
rm -fr *
cd ..
rmdir LAB1


