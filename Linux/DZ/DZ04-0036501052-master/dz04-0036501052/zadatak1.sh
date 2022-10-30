#!/bin/bash

# Zadatak 1.a
touch $(whoami)

# Zadatak 1.b
echo "USER: $(whoami)" >> $(whoami)
echo "HOME DIR: /home$(whoami)" >> $(whoami)
echo "SHELL: $SHELL" >> $(whoami)
echo "GROUPS: $(id -nG)" >> $(whoami)
echo "PRIMARY GROUP: "$(id -ng)	"" >> $(whoami) 

# Zadatak 1.c
cat $(whoami)
rm $(whoami) 

# Ako si znao za $SHELL, kako nisi znao i za $USER, $HOME i $GROUPS, recimo? :)
# No, nema veze, snašao si se.
