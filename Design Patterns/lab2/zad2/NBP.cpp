#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


int gt_int(const void* l, const void* r){
    int p = *(const int*) l;
    int q = *(const int*) r;
    int res = q - p;
    
    if(res > 0){
        return 1;
    }
    else if (res < 0){
        return -1;
    }
    else{
        return 0;
    }
}

int gt_char(const void* l, const void* r){
    const char p  = *(const char*) l;
    const char q = *(const char*) r;
    char res = q - p;
    
    if(res > 0){
        return 1;
    }
    else if (res < 0){
        return -1;
    }
    else{
        return 0;
    }
}


int gt_str(const void* l, const void* r){
    const char* p  = *(const char**) l;
    const char* q = *(const char**) r;
   return strcmp(q,p);
}





const void* mymax(const void *base, size_t nmemb, size_t size,
                  int (*compar)(const void *, const void *)){ 
  const void* max = base;                    
  for(int i = 0; i < nmemb; ++i ){
      const void* p = base + size*i;
      if(compar(max, p) > 0){
          max = p;
      }
  }
  return max;
};


int main(){
    int arr_int[] = { 1, 3, 5, 7, 4, 6, 9, 2, 0 };
    char arr_char[]="Suncana strana ulice";
    const char* arr_str[] = {
        "Gle", "malu", "vocku", "poslije", "kise",
        "Puna", "je", "kapi", "pa", "ih", "njise"
    };  


    const void* x = mymax(arr_int, 9, sizeof(arr_int[0]), gt_int);
    printf("max_int : %d\n",*(int*) x);

    const void* y = mymax(arr_char, 20, sizeof(arr_char[0]), gt_char);
    printf("max_char : %c\n",*(char*) y);

    const void* z = mymax(arr_str,11, sizeof(char*), gt_str);
    printf("max_str : %s",*(char**) z);
}