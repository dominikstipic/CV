#include <stdio.h>
#include <stdlib.h>

class Unary_Function{
    private:
        int lower_bound;
        int upper_bound;

    public:
        Unary_Function(int la, int ub): lower_bound(la), upper_bound(ub){};
        //virtual abstract function
        virtual double value_at(double x) = 0; 
        virtual double negative_value_at(double x){
            return -value_at(x);
        }
        void tabulate(){
            for(int x = lower_bound; x <= upper_bound; ++x){
                printf("f(%d) = %lf\n",x,value_at(x));
            }
        }
        static bool same_function_for_ints(Unary_Function *f1, Unary_Function *f2, double tolerance){
            if(f1->lower_bound != f2-> lower_bound) return false;
            if(f1-> upper_bound != f2->upper_bound) return false;

            for(int x = f1->lower_bound; x <= f1->upper_bound; ++x){
                double delta = f1 -> value_at(x) - f2->value_at(x);
                if(delta < 0) delta = -delta;
                if(delta > tolerance) return false;
            }
            return true;
        }
};

class Square : public Unary_Function{
    public:
        Square(int lb, int ub) : Unary_Function(lb,ub){};
        virtual double value_at(double x){
            return x*x;
        } 
};

class Linear : public Unary_Function{
    private:
        double a;
        double b;
    public:
        Linear(int lb, int ub, double a_coef, double b_coef) : Unary_Function(lb, ub), a(a_coef), b(b_coef){};
        virtual double value_at(double x){
            return a*x + b;
        }
};


int main(){
    Unary_Function *f1 = new Square(-2,2);
    f1-> tabulate();
    printf("-------\n");
    Unary_Function *f2 = new Linear(-2,2,5,-2);
    f2 -> tabulate();
    printf("f1 == f2: %s\n", Unary_Function::same_function_for_ints(f1,f2,1E-6) ? "DA" : "NE");
    delete f1;
    delete f2;
    return 0;
}


/*
          | -----------------|
          |  Unary_Function  |
          | -----------------| 
                   ^
                   |
                   |
        |-----------------------|
        |                       |
        |                       |
        |                       |
  | -----------------|   |--------------|     
  |       Square     |   |    Linear    |
  | -----------------|   |--------------|

*/


