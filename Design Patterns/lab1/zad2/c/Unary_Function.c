# include <stdio.h>
# include <stdlib.h>
# define VALUE_AT_FUN 0
# define NEGATIVE_VALUE_AT_FUN 1

struct VTable;

struct Unary_Function{
    int lower_bound;
    int upper_bound;
    struct VTable* vtab;
    void (*dtor)(struct Unary_Function*);
    void* derived;
};

struct VTable{
    double (*value_at)(struct Unary_Function*, double);
    double (*negative_value_at)(struct Unary_Function* , double);
};

double value_at(struct Unary_Function* ,double);
double negative_value_at(struct Unary_Function*, double);
void destructor(struct Unary_Function*);

void create_Unary_Function(struct Unary_Function* self, int lower_bound, int upper_bound){
    self -> lower_bound = lower_bound;
    self -> upper_bound = upper_bound;
    self -> dtor = destructor;
    self -> vtab = (struct VTable*) malloc(sizeof(struct VTable));
    self -> vtab -> value_at = value_at;
    self -> vtab -> negative_value_at = negative_value_at; 
    self -> derived = self;
}

struct Unary_Function* construct_Unary_Function(int lower_bound, int upper_bound){
    struct Unary_Function* uf = (struct Unary_Function*) malloc(sizeof(struct Unary_Function));
    create_Unary_Function(uf, lower_bound, upper_bound);
    return uf;
}

void delete(struct Unary_Function* self){
    self -> dtor(self);
}

void destructor(struct Unary_Function* self){
    free(self -> vtab);
    free(self -> derived);
}

void tabulate(struct Unary_Function* self){
    int x = self -> lower_bound;
    int y = self -> upper_bound;
    for(int i = x; i <= y; ++i){
        printf("f(%d)=%lf\n", i, self -> vtab -> value_at(self, i));
    }
};

int same_functions_for_ints(struct Unary_Function *f1, struct Unary_Function *f2, double tolerance) {
      if(f1->lower_bound != f2->lower_bound) return 0;
      if(f1->upper_bound != f2->upper_bound) return 0;

      for(int x = f1->lower_bound; x <= f1->upper_bound; x++) {
        double delta = f1 -> vtab -> value_at(f1, x) - f2 -> vtab -> negative_value_at(f2, x);
        if(delta < 0) delta = -delta;
        if(delta > tolerance) return 0;
      }
      return 1;
    };

double value_at(struct Unary_Function* self, double x){return 0;}

double negative_value_at(struct Unary_Function* self, double x){
    return -value_at(self, x);
};