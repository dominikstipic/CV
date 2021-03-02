class ApstraktnaAkcija():
    def __init__ (self,odrediste=None):
        self.odrediste = odrediste

    def procesuiraj(self,podaci):
        pass


class Logger(ApstraktnaAkcija):
    def __init__ (self,odrediste=None):
        import os
        assert odrediste != None
        assert os.path.isfile(odrediste), "Putanja do datoteke nije ispravna"
        self.odrediste = odrediste

    def procesuiraj (self,podaci):
        import datetime
        d = datetime.datetime.now()
        with open(self.odrediste,"w") as fp:
            for x in podaci:
                fp.write(str(x)+",")
            fp.write(" : " + str(d))
            print("Podaci zapisani u: {}".format(self.odrediste))


class Stats(ApstraktnaAkcija):
    def procesuiraj(self,podaci):
        assert type(podaci) == list, "Ne kompatibilan tip podataka"
        import numpy as np

        suma    = sum(podaci)
        prosijek = suma / len(podaci)
        medijan =  np.median(podaci)

        print("suma     : {}".format(suma))
        print("prosijek : {}".format(prosijek))
        print("medijan  : {}".format(medijan))

