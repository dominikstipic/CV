
#include "myfactory.h"
#include <stdio.h>
#include <stdlib.h>

typedef char const* (*PTRFUN)();

struct Animal{
  PTRFUN* vtable;
  // vtable entries:
  // 0: char const* name(void* this);
  // 1: char const* greet();
  // 2: char const* menu();
};

void animalGetName(struct Animal* self){
    char const* name = self -> vtable[2]();
    printf("%s",name);
}

void animalPrintGreeting(struct Animal* self){
     char const* p = self -> vtable[1]();
    printf("%s",p);
}

void animalPrintMenu(struct Animal* self){
     char const* p = self -> vtable[2]();
    printf("%s",p);
}

struct Animal* createAnimal(){
    return (struct Animal*) malloc(sizeof(struct Animal));
}




int main(void){
  struct Animal* p1=(struct Animal*)myfactory("parrot", "Modrobradi");
  struct Animal* p2=(struct Animal*)myfactory("tiger", "Straško");
  if (!p1 || !p2){
    printf("Creation of plug-in objects failed.\n");
    exit(1);
  }

  animalPrintGreeting(p1);//"Sto mu gromova!"
  animalPrintGreeting(p2);//"Mijau!"

  animalPrintMenu(p1);//"brazilske orahe"
  animalPrintMenu(p2);//"mlako mlijeko"

  free(p1); free(p2); 
  return 0;
}