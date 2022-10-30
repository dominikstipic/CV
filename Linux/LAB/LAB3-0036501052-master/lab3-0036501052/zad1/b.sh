#!/bin/bash


input=$(sudo cat /dev/stdin)
cd /home/studenti

for i in $input 
do
    sudo -u stjepan sudo mkdir $i
    sudo -u stjepan sudo adduser -s "/bin/bash" -m "./$i" $i
    sudo -u stjepan sudo chmod u+rwx,g-rwx,o-rwx $i
    sudo -u stjepan sudo chown $i $i
    cd $i
	cp ../skeleton/* .
	  
	

    cd ..	
done
	
