# include <stdio.h>
# include "Unary_Function.c"
# include "Square.c"
# include "Linear.c"

int main(void){
    struct Unary_Function* f1 = construct_Unary_Function(-2,2);
    struct Unary_Function* f2 = constructSquare(-2,2);
    struct Unary_Function* f3 = constructLinear(-4,2,2,2);

    tabulate(f1);
    printf("-------\n");
    tabulate(f2);
    printf("-------\n");
    tabulate(f3);

    delete(f1);
    delete(f2);
    delete(f3);
}
