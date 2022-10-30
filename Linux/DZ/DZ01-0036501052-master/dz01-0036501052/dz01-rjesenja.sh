#!/bin/bash

# Zadatak 1
cat ~/.bash_logout

# Zadatak 2
ls -rS

# Zadatak 3.a
mkdir -p /tmp/OKOSLtjedan/{ponedjeljak,utorak,srijeda,cetvrtak,petak,.subota}
# Zadatak 3.b
ls .
# ovdje je bio zadatak ispisati naziv trenutnog radnog direktorija (ne njegov sadržaj)
# to se radi sljedećom naredbom: pwd
# Zadatak 3.c
touch /tmp/OKOSLtjedan/.subota{predavanja,labosi}
for i in {1..8}
do
touch /tmp/OKOSLtjedan/.subota/zadaca$i
done
# svaka čast na petlji, ali kako bi to riješio da pišeš direktno u terminalu, tj. bez petlje? 
# u tom slučaju primijeni sljedeću metodu:
# touch <path>/{predavanja,labosi,zadaca{1..8}} 
# Zadatak 3.d
cp -r /tmp/OKOSLtjedan/.subota /tmp/OKOSLtjedan/pondedjeljak
# Zadatak 3.e
ls -R /tmp/OKOSLtjedan


# Zadatak 4.a
ln -s /var Varionica
# Zadatak 4.b
du -shLc Varionica | tail -1
# Zadatak 4.c
rm Varionica

# Zadatak 5
df -h / 

# Zadatak 6
file /bin/bash
file /etc/passwd
file /boot
# ovdje se očekuje jednolinijsko rješenje :) 
# file /bin/bash /etc/passwd /boot


