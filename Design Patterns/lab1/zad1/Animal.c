#include "string.h"
#include <stdio.h>

struct Animal{
    char* name;
    char const* (**PTRFUN)();
};


void animalPrintGreeting(struct Animal* p){
   printf("%s\n", p->PTRFUN[0]());
}

void animalPrintMenu(struct Animal* p){
    printf("%s\n", p->PTRFUN[1]());
}