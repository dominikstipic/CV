#include <stdio.h>
#include <stdlib.h>


struct Parrot{
    char const* name;
    struct Animal* parent;
    char const* (*(*ptr)[3])();
};

char const* parrotName();
char const* parrotGreet();
char const* parrotMenu();

char const* (*fun[3])() = {parrotName, parrotGreet, parrotMenu};

 void* create(char const* name){
     struct Parrot* parrot = (struct Parrot*) malloc(sizeof(struct Parrot));
     //parrot -> parent = (struct Animal*) malloc(sizeof(struct Animal));
     parrot -> name = name;
     parrot -> ptr = &fun;
     return parrot;
 }

// void parrotName(struct Parrot* self){
//     printf("%s",self -> name);
// }

// char const* parrotName(struct Parrot *p){
//      return p->name;
// }

char const* parrotName(){
      return "ime";
}

char const* parrotGreet(){
    return "Sto mu gromova!";
}

char const* parrotMenu(){
   return "brazilske orahe";
}

