#!/bin/bash

# Zadatak 2.a
# Weakling se ne nalazi u grupi sudo pa zato tu naredbu nemoze ni koristiti
addusser weakling
su weakling

#nemože koristiti sudo
sudo apt install hollywood

#odjava
exit

# dodajemo weaklinga u sudo grupu. Tu operaciju radi korisnik koji može koristiti sudo
adduser weakling sudo
su weakling 
sudo apt install hollywood

# Zadatak 2.b
exit
deluser weakling
