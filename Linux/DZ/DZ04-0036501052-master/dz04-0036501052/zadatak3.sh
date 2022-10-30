#!/bin/bash

cd /root
# Permission denied

sudo cd /root
# cd command not found


touch /tmp/spremnik
echo "$(whoami)" > /tmp/spremnik
# spremam valstito ime

sudo su
# prelazimo u "super usera"

cd /root

#vraćamo se nazad
su "$(cat /tmp/moje)"

ls -a
rm /tmp/spremnik
#običan korisnik nema ovlasti za čitanje sadržaja root direktorija

#Bonus
# Sudo naredba kao argument očekuje izvršnu datoteku. Cd je ugrađena funkcionalost terminala i zbog toga sudo nemože pronaći izvršnu datoteku imena cd.


# Točno! Malo si brljao s imenima fajlova (/tmp/moje, /tmp/spremnik), ali vidim što si htio, nije problem.
# Dobra zadaća!

# Bodovi: 6/6
# Ispravio leonard.volaric.horvat@kset.org
