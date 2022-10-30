#!/bin/bash

# Zadatak 1.a
ls /usr/share/ | egrep ^gtk

# Čini se da si pobrkao zamjenske znakove (wildcards) i regularne izraze (regex).
# Wildcardovi su svojstvo ljuske tj. prije izvođenja naredbe ih ljuska prvo zamijeni sa stvarnim
# vrijednostima, a ^ ne znači ništa posebno tj. interepretira se kao taj znak.
# Regularni izrazi su konstrukt za sebe i ljuska ih samostalno ne interpretira, kao ni naredba ls.
# Rješenje: ls -d /usr/share/gtk*


#Zadatak 1.b
ls /usr/share/ | egrep .*[0-9].*[0-9].*

# Opet, pobrkao si wildcardove i regexe. U wildcardovima je točka običan znak. 
# Rješenje: ls /usr/share/ | egrep *[0-9]*[0-9]*

#Zadatak 2.a
cat /usr/share/dict/words | grep [0-9]
#Zadatak 2.a
cat /usr/share/dict/words | egrep ^i.*[A-Z].+$
#Zadatak 2.c
cat /usr/share/dict/words | grep ^[^aeiouAEIOU]$
# Zadtak 2.d
cat /usr/share/dict/words | grep .*[aeiou][aeiou].*

# Točno, ali moglo je malo ljepše:
# egrep -i '[aeiou]{2}' /usr/share/dict/words

# Zadtak 2.e
 cat /usr/share/dict/words | grep -c .*ening$
# Zadtak 2.f
 cat /usr/share/dict/words | grep -c \'s$
# Zadtak 2.e
cat /usr/share/dict/words | egrep -c [A-Z]$

# Općenito, zašto cat pa pipe na grep? grep sasvim dobro razumije fajlove kao argumente :)


# Zadatak 3.a
cat /usr/share/dict/words | sed -r "s:'s$:s:"

# Zadatak 3.b
cat /usr/share/dict/words | egrep .+word.* |sed -r "s:(.+)(word)(.*):\2\1\3:"

# Interesantan pristup. Ali ovaj grep ti samo smeta. Nema potrebe za njim jer će sed ionako promjene
# raditi samo nad linijama koje sadrže niz "word".
# Dodatni problem s tvojim pristupom jest da je beskoristan za uređivanje fajlova inline jer ne radi
# nad fajlom direktno, nego nad outputom grepa. sed, baš kao i grep, sasvim dobro razumije fajlove kao argumente!
# Ipak, regex je dobar, pa ne uzimam za zlo.


# Zadatak 3.c
cat /usr/share/dict/words | sed  -r "s:([A-Z])(.*)([a-z]):\L\1\E\2\U\3:"

# Skoro! Ovdje ti nedostaje znak za početak i za kraj linije. U zadatku je definirano da
# se tražene transformacije rade samo nad linijama koje započinju velikim slovom.
# Točno rješenje: sed -r "s/^([A-Z])(.*)([a-zA-Z])$/\L\1\2\U\3/" /usr/share/dict/words

# Solidna zadaća, ali malo si brljao s regexima i wildcardovima i nepotrebnim pipeovima.

# Bodovi: 6/8
# Ispravio leonard.volaric.horvat@kset.org
