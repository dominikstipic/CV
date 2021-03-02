#include <stdio.h>

class B{
public:
  virtual int prva()=0;
  virtual int druga()=0;
};

class D: public B{
public:
  virtual int prva(){return 0;}
  virtual int druga(){return 42;}
};



/*
Virtualne metode možemo pozivati i bez korištenja sim. imena.
Vtab se nalazi na početku koda svakog objekta.
*/
/*
Potrebno je napisati funkciju koja prima pokazivač pb na objekt razreda B te ispisuje povratne vrijednosti dvaju metoda, 
ali na način da u kodu ne navodimo simbolička imena prva i druga.
Zadatak pokušajte riješiti korištenjem samo jednog operatora pretvorbe (engl. cast operator).
*/

void fun(B* p){
    /* p              -> adresa strukture, odnosno njenog prvog elementa -> VTAB
     * (*p)           -> adresa vtab
     * (*p)[0]        -> adresa prve virtualne metode
     * *((p)[0])(...) -> poziv funkcije
     */
    
    typedef int (*funPtr)();   // function pointer
    funPtr** vtabAdr = (funPtr**) p; // adresa vtab
    funPtr* vtab = *vtabAdr; // vtablica, adresa metode 0
    funPtr fun1 = *vtab;
    funPtr fun2 = *(vtab + 1);

    int (**vadr)() = (int (**)())p;
    int (*vt)() = *vadr;
    

     
    printf("%d~n", fun1());
    printf("%d",fun2());



}

int main(void){
    D d;
    fun(&d);
    return 0;
}