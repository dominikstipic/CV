#!/bin/bash

#   Za sva pitanja, prijedloge i dileme oko ovog zadatka osjećajte se slobodno
#   slati mail na darko.janekovic@fer.hr ili okosl@kset.org
#
#   S ljubavlju napisao,
#   Darko Janeković, 17.12.2018


usage="
Ova skripta postavlja kulise za zadatak 6. iz druge laboratorijske vježbe
Osnova korištenja operacijskog sustava Linux. Skripta će nas premjestiti
u \tmp direktorij i tamo napraviti direktorij OKOSL.

U direktoriju OKOSL ćemo napraviti downloads direktorij unutar kojeg ćemo pokrenuti
skriptu generate_files.sh. Skripta generate_files.sh će napuniti direktorij umjetnim
datotekama nad kojima ćete moći vježbati sortiranje.

Na kraju ćemo kopirati skriptu sort.sh iz trenutnog direktorija u /tmp/OKOSL kako
biste vi mogli samo otici u /tmp/OKOSL i pokrenuti bash sort.sh.

Ako je ovo bilo konfuzno ispisivat ću svaki korak koji radim pa će možda biti
jasnije."

printf "%s\n" "$usage"

echo "Radim direktorij /tmp/OKOSL/downloads..."
mkdir -p /tmp/OKOSL/downloads
if [[ -d /tmp/OKOSL/downloads ]]
then
    echo "Napravljen je direktorij /tmp/OKOSL/downloads"
else
    echo "Nešto ne valja, /tmp/OKOSL/downloads ne postoji, odustajem..."
    exit 1
fi

echo "kopiram skriptu generate_files.sh u /tmp/OKOSL/ ..."
if [[ -e ./generate_files.sh ]]
then
    cp ./generate_files.sh /tmp/OKOSL/.
else
    echo "./generate_files.sh ne postoji u trenutnom direktoriju, wgetam file"
    wget https://marvin.kset.org/~okosl/lab2/generate_files.sh -O /tmp/OKOSL/generate_files.sh
fi

# odi u /tmp/OKOSL/, pokreni skriptu i vrati pwd kakav je bio prije
cd /tmp/OKOSL/ && bash ./generate_files.sh && cd -

if [[ -e ./sort.sh ]]
then
    echo "kopiram sort.sh u /tmp/OKOSL/. ..."
    cp ./sort.sh /tmp/OKOSL/.
else
    echo "sort.sh ne postoji u trenutnom direktoriju. Pa ne mogu kopirati"
fi

echo "Gotovo! Sretno sa zadatkom :)"
