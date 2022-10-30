#!/bin/bash

# Zadatak 1.a
touch personal_info.dat
echo 0036501052 > personal_info.dat
cat personal_info.dat
# Zadatak 1.b
cat >> personal_info.dat << END
Dominik
Stipić
END
cat personal_info.dat
# Zadatak 1.c
echo okosl | tee -a personal_info.dat


# Zadatak 2.a
cd /tmp/OKOSL
sort -h korisnici.dat > korisnici.sortirano.dat
cat korisnici.sortirano.dat
# Zadatak 2.b
cut -f2 -d: korisnici.dat | sort -h | uniq -iu > jedinstveni_korisnici.dat
cat jedinstveni_korisnici.dat
# Zadatak 2.c 
cut -f2 -d: korisnici.dat | sort -h | uniq -id > nejedinstveni_korisnici.dat
cat nejedinstveni_korisnici.dat


# Zadatak 3.a
grep ping /usr/share/dict/words | wc -w

# Točno, ali mogao si uštediti ovaj wc, korištenjem grepove zastavice "-c" (count)


# Zadatak 3.b
grep you /usr/share/dict/words > yous.dat
cat yous.dat
# Zadatak 3.c
cat yous.dat | wc -w
# Zadatak 3.d
cat yous.dat > /tmp/help.txt && cat /tmp/help.txt >> yous.dat
# Moglo je i bez pomoćne datoteke, npr. koristeći naredbu tee:
# cat yous.dat | tee --append yous.dat

# Zadatak 4.a
find /tmp -name jedinstveni_korisnici.dat 2> /dev/null
# Zadatak 4.b
updatedb
locate jedinstveni_korisnici.dat


# Zadatak 5.a
ls -l | grep lis
# Zadatak 5.b
ls -l | grep lis | tr -s " " | sort -hk7
# Zadatak 5.c
ls -l | grep lis | tr -s " " | sort -hk7 > /tmp/sortiran_home.txt
cat /tmp/sortiran_home.txt
test -f /tmp/sortiran_home.txt && echo "datoteka postoji"


# Izvrsna zadaća! Sve je točno; nemam posebnih komentara.
# Bodovi: 8/8
# Ispravio leonard.volaric.horvat@kset.org
