#include <stdlib.h>

struct Linear{
    struct Unary_Function* parent;
    double a;
    double b;
    void (*dtor)(struct Unary_Function*);
};

double linear_value_at(struct Unary_Function* ,double);
void destructorLinear(struct Unary_Function*);

struct Unary_Function* constructLinear(int lb, int ub, double a_coef, double b_coef){
    struct Linear* lin = (struct Linear*) malloc(sizeof(struct Linear));
    struct Unary_Function* parent = (struct Unary_Function*) malloc(sizeof(struct Unary_Function));
    create_Unary_Function(parent, lb, ub);
    lin -> parent = parent;
    parent -> derived = lin;
    parent -> dtor = destructorLinear;
    parent -> vtab -> value_at = linear_value_at;
    lin -> a = a_coef;
    lin -> b = b_coef;
    return parent;
}

double linear_value_at(struct Unary_Function* self ,double x){
    struct Linear* lin = (struct Linear*) self -> derived;
    double a = lin -> a;
    double b = lin -> b;
    return a*x + b;
}

void destructorLinear(struct Unary_Function* self){
    free(self -> vtab);
    free(self -> derived);
    free(self);
}