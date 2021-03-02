#include <stdlib.h>

typedef char const* (*PTRDOG)();
char const* dogGreet(void);
char const* dogMenu(void);

PTRDOG arrDog[2] = {dogGreet, dogMenu};

//void ConstructDog(self, String name)
void constructDog(struct Animal* dog, char *name){
    dog->name = name;
    dog->PTRFUN = arrDog;
}

//new Dog(String name)
struct Animal* createDog(char* name){
    struct Animal* dog = (struct Animal*) malloc(sizeof(struct Animal));
    constructDog(dog,name);
    return dog;
}

char const* dogGreet(void){
    return "vau!";
}

char const* dogMenu(void){
    return "kuhanu govedinu";
}