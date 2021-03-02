from Akcije import ApstraktnaAkcija
from Izvori import ApstraktniIzvor

class SlijedBrojeva:
    def __init__(self,izvor):
        self.izvor = izvor
        self.kolekcija = []
    
    def kreni(self,akcije):
        import time
        while(True):
            x = self.izvor.preuzmi_podatak()
            print("podatak: {}".format(x))
            if x == -1:
                break
            self.kolekcija.append(x)
            time.sleep(1)
    
        for a in akcije:
            a.procesuiraj(self.kolekcija)


