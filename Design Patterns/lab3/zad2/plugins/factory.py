import imp
import os

def myfactory(moduleName):
    fp,path,description = imp.find_module("plugins/"+moduleName)
    module = imp.load_module(moduleName,fp,path,description)
    fp.close()
    return getattr(module,moduleName.title())
