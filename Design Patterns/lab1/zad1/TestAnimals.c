# include <stdio.h>
# include "Animal.c"
# include "Dog.h"
# include "Cat.h"
# include "string.h"

void testAnimals(){
    struct Animal* p1 = createDog("Hamlet");
    struct Animal* p2 = createCat("Ofeija");
    struct Animal* p3 = createDog("Polonije");

    animalPrintGreeting(p1);
    animalPrintGreeting(p2);
    animalPrintGreeting(p3);

    animalPrintMenu(p1);
    animalPrintMenu(p2);
    animalPrintMenu(p3);

    free(p1); free(p2), free(p3);
}

void lajte(struct Animal** psi, int n){
     for(int i = 0; i < n; ++i){
        animalPrintGreeting(psi[i]);
    }
}

void dogs(int n){
    struct Animal** psi = (struct Animal**) malloc(n * sizeof(struct Animal));

    for(int i = 0; i < n; ++i){
        char str[10];
        psi[i] = createDog("pas");
    }

    lajte(psi,n);

    for(int i = 0; i < n; ++i){
        free(psi[i]);
    }   
    free(psi);
}

int main(){
    int n = 5;
    dogs(n);
    testAnimals();
    return 0;
}

