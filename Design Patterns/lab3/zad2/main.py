import os
from plugins.factory import myfactory


def test():
    pets=[]
    # obiđi svaku datoteku kazala plugins 
    for mymodule in os.listdir('plugins'):
        moduleName, moduleExt = os.path.splitext(mymodule)
    # ako se radi o datoteci s Pythonskim kodom ...
    if moduleExt=='.py':
        # instanciraj ljubimca ...
        ljubimac=myfactory(moduleName)('Ljubimac '+str(len(pets)))
        # ... i dodaj ga u listu ljubimaca
        pets.append(ljubimac)
    
    return pets







def printGreeting(ime,kaze):
  print("{} kaze {}".format(ime,kaze))

def printMenu(ime,menu):
  print("{} {}".format(ime,menu))

  # ispiši ljubimce
for pet in test():
    name = pet.name
    printGreeting(name,pet.greet())
    printMenu(name,pet.menu())