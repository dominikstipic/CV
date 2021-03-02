#include <stdlib.h>

typedef char const* (*PTRCAT)();
char const* catGreet(void);
char const* catMenu(void);

PTRCAT arrCat[2] = {catGreet, catMenu};

//void ConstructCat(self, String name)
void constructCat(struct Animal* cat, char *name){
     cat->name = name;
     cat->PTRFUN = arrCat;
}

//new Cat(String name)
struct Animal* createCat(char* name){
    struct Animal* cat = (struct Animal*) malloc(sizeof(struct Animal));
    constructCat(cat,name);
    return cat;
}

char const* catGreet(void){
    return "mijau!";
}

char const* catMenu(void){
    return "konzerviranu tunjevinu";
}