#!/bin/bash




touch Top10
cat > Top10 << END
Linux Mint 17.2
Ubuntu 15.10
Debian GNU/Linux 8.2
Mageia 5
Fedora 23
openSUSE Leap 42.1
Arch Linux
CentOS 7.2-1511
PCLinuxOS 2014.12
Slackware Linux 14.1
FreeBSD
END

cat Top10
echo "--------\n"

#briše distribucije bez verzije
sed -i '/^[^0-9]*$/d' Top10 
cat Top10

echo "--------\n"

# verzija prebacena na pocetak linije
sed -ri "s:(.*)[ ]([0-9].*[-.]?.*[0-9]?):\2 \1:" Top10
cat Top10

echo "--------\n"

#  sva slova su promijenjena u mala
sed -ri "s:(.*):\L\1:" Top10 
cat Top10

echo "--------\n"

# svi samoglasnici prebaceni u velika slova
sed -ri "s:([aeiou]):\U\1:g" Top10 
cat Top10

echo "--------\n"

# sortirati datoteku po numeričkoj vrijednosti na početku
sort -o Top10 -n Top10
cat Top10



