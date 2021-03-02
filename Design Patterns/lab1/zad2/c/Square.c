#include <stdlib.h>

struct Square{
    struct Unary_Function* parent;
    void (*dtor)();
};

void destructorSquare(struct Unary_Function*);

double square_value_at(struct Unary_Function* self ,double x){
    return x * x;
}

struct Unary_Function* constructSquare(int lb, int ub){
    struct Square* square = (struct Square*) malloc (sizeof(struct Square));
    struct Unary_Function* parent = (struct Unary_Function*) malloc(sizeof(struct Unary_Function));
    create_Unary_Function(parent, lb, ub);
    square -> parent = parent;
    square -> parent -> derived = square;
    square -> parent -> dtor = destructorSquare;
    parent -> vtab -> value_at = square_value_at;
    return parent;
}

void destructorSquare(struct Unary_Function* self){
    free(self -> vtab);
    free(self -> derived);
    free(self);
}