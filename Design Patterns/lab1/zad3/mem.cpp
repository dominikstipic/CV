# include <stdio.h>
# include <stdlib.h>

class CoolClass{
public:
  virtual void set(int x){x_=x;};
  virtual int get(){return x_;};
private:
  int x_;
};

class PlainOldClass{
public:
  void set(int x){x_=x;};
  int get(){return x_;};
private:
  int x_;
};


int main(void){
    printf("CoolClass:%u B\n", sizeof(CoolClass));   // 16
    printf("PlainOldClass:%u B \n", sizeof(PlainOldClass));  // 4


    /* sizeof
     *   computes the required memory storage of its operand.
     *   when applied to the address of an array, the result is the number
     *   of bytes required to store the entire array.
     */

    /*
     * PlainOldClass -> struct{int x} -> sizeof(int) = 4
     * 
     * CoolClass -> struct{int x, vptr}
     *           -> mem. lokacija = 64 bit = 8 B
     *           -> sizeof(int) + sizeof(pointer to v.t.) = 4 + 8 = 12
     *           -> padding do 16 B 
     *            
     *
     *    
     */

    return 1;
}

