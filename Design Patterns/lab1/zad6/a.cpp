#include <stdio.h>

class Base{
public:
  Base() {
    printf("Base\n");
    metoda();
  }

  virtual void virtualnaMetoda() {
    printf("ja sam bazna implementacija!\n");
  }

  void metoda() {
    printf("Metoda kaze: ");
    virtualnaMetoda();
  }
};

class Derived: public Base{
public:
  Derived(): Base() {
    printf("Derived\n");
    metoda();
  }
  virtual void virtualnaMetoda() {
    printf("ja sam izvedena implementacija!\n");
  }
};

int main(){
  Derived* pd=new Derived();
  pd->metoda();
}

/* u Derived() konstruktoru najprije se poziva Base() or super()
 * Base() -> Base(struct Base* self) 
      * self -> vtab -> metoda = &metoda
      * self -> vtab -> metoda pokazuje na Base :: metoda()
 * poziva se metoda() tj. self -> vtab -> metoda();
 *  self -> vtab -> metoda pokazivac pokazuje na Base :: metoda()
 * vraćamo se iz Base()
 * poslije super() izmjenjuje se vtab
 * self -> vtab -> metoda = Derived :: metoda()
 * poziva se Derived :: metoda()
*/