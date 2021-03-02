#define VALUE_AT_FUN 0
#define NEGATIVE_VALUE_AT_FUN 1

struct Unary_Function{
    int lower_bound;
    int upper_bound;
    double (**VPTR)(void*, double);
    void (*dtor)(struct Unary_Function*);
};

double value_at(void* ,double);
double negative_value_at(void*, double);
void delete(struct Unary_Function*);
void create_Unary_Function(struct Unary_Function*, int, int);