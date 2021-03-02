#include "main.c"
#include <stdio.h>
#include <stdlib.h>




struct Tiger{
    char const* name;
    struct Animal* parent;
    char const* (*(*ptr)[3])();
};

char const* tigerName();
char const* tigerGreet();
char const* tigerMenu();

char const* (*fun[3])() = {tigerName, tigerGreet, tigerMenu};

 void* create(char const* name){
     struct Tiger* tiger = (struct Tiger*) malloc(sizeof(struct Tiger));
     tiger -> parent = (struct Animal*) malloc(sizeof(struct Animal));
     tiger -> name = name;
     tiger -> ptr = &fun;
     return tiger;
 }

// void parrotName(struct Parrot* self){
//     printf("%s",self -> name);
// }

// char const* parrotName(struct Parrot *p){
//      return p->name;
// }

char const* tigerName(){
      return "ime";
}

char const* tigerGreet(){
    return "Sto mu gromova!";
}

char const* tigerMenu(){
   return "brazilske orahe";
}

