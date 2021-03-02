#include <stdlib.h>
#include <string.h>
#include <vector>
#include <stdio.h>
#include<iostream> 
#include<vector>



template <typename Iterator, typename Predicate>
Iterator mymax(Iterator cur, Iterator last, Predicate pred){
    Iterator max = cur;
    while(cur != last){
        if(pred(*max,*cur)){
            max = cur;
        }
        ++cur;
    }
    return max;
}


template <typename T>
bool comparator( T l,  T r){
    return l < r;
}

bool gt_str_inv(const char* l, const char* r){
    return strcmp(r,l);
}

bool gt_str(const char* l, const char* r){
    return strcmp(l,r);
}

int gt_int( int l, int r){
    int res = l - r;
    
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

int main(){
    int arr_int[] = { 1, 3, 5, 7, 4, 6, 9, 2, 0 };
    char arr_char[]="Suncana strana ulice";
    const char* arr_str[] = {
        "Gle", "malu", "vocku", "poslije", "kak,",
        "Puna", "je", "kapi", "pa", "ih", "njiše"
    };  
    std::vector<int> vec = {1,2,3,4};



    const int* maxint = mymax( &arr_int[0],
                              &arr_int[sizeof(arr_int)/sizeof(*arr_int)], comparator<int>);
    std::cout <<*maxint <<"\n";

    const char* max_char = mymax( &arr_char[0],
                              &arr_char[sizeof(arr_char)/sizeof(*arr_char)], comparator<char>);
    std::cout <<*max_char <<"\n";

    const char** max_str = mymax( arr_str,
                              &arr_str[sizeof(arr_str)/sizeof(*arr_str)], gt_str_inv);
    std::cout <<*arr_str <<"\n";

    const char** max = mymax( arr_str,
                              &arr_str[sizeof(arr_str)/sizeof(*arr_str)], gt_str);
    std::cout <<*max <<"\n";

     std::vector<int>::iterator maxVec = mymax( vec.begin(),
                                                vec.end(), comparator<int>);
    std::cout << *maxVec <<"\n";


    
    return 0; 

   

   
}