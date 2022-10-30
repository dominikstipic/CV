#!/bin/bash


sudo useradd -s"/bin/bash" -M stjepan 
p1="/usr/sbin/adduser"
p2="/usr/sbin/deluser"
p3="/bin/chmod"
p4="/usr/sbin/groupadd"
p5="/usr/sbin/groupdel"
p6="/bin/mkdir"
p7="/bin/chgrp"
p8="/bin/chown"
p9="/bin/ls"

echo "stjepan kreiran"
read

echo stjepan ALL=NOPASSWD:$p1,$p2,$p3,$p4,$p5,$p6,$p7,$p8,$p9 | sudo EDITOR="tee -a" visudo

echo "stvaram dir studenti"
read

cd /home
sudo -u stjepan sudo groupadd studenti
sudo -u stjepan sudo mkdir studenti
sudo -u stjepan sudo chmod -R -w studenti
cd studenti

echo "privilegije"
read

sudo -u stjepan sudo mkdir studenti-shared
sudo -u stjepan sudo chmod u=rwx,g+rwx,o-rwx studenti-shared
#vlasniku i ostalima zabrani read,write,execute
sudo -u stjepan sudo chmod +t studenti-shared
#in-memory direktorij
sudo -u stjepan sudo chgrp studenti studenti-shared
#studenti su vlasnici dir-a

sudo -u stjepan sudo ls -l studenti-shared

echo "skelet"
read

sudo -u stjepan sudo mkdir skeleton
cd skeleton
sudo -u stjepan sudo mkdir Desktop Download Documents Media
sudo -u stjepan ln -s ../studenti-shared Shared
cd ..

echo "pokrećem skriptu"
read

cd /home/doms/git/LAB3-0036501052/zad1
cat studenti | ./b.sh




