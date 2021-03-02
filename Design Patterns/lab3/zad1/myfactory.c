#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Linux support 2 classes of libs:
 *      static libraries
 *          * Routines that are compiled and linked directly ino your program
 *          * Binary code contains lib code and client code
 *          * If library changes the executable must be recompiled
 *      dynamic or shared libraries
 *          * Routins that are loaded into app at runtime
 *          * The Library does not become part of executable
 *          * A lot of code can share same library and we dont need to recompile 
 *            executable
 * 
 *          dynamically linked libs
 *          dynamically loaded libs
 * 
 * dlopen:
 *      Loads the dynamic library file named by the null-terminated string filename
 *      and returns an opaque handle for he dynamic lib
 * dsym:
 *      * Takes a handle of a dynamic library and the null-terminated symbol name,
 *      * Returns the address where the symbol is loaded into memory 
 * 
 * */

void* myfactory(char const* libname, char const* ctorarg){
    void *handle;
    char *error;
    struct Animal* (*factory)(char const*);
    
    handle = dlopen(libname,RTLD_LAZY);
    if (!handle) {
       return NULL;
    }
    dlerror();

    *(void **) (&factory) = dlsym(handle,ctorarg);
    dlclose(handle);
    return (*factory)(ctorarg);

}