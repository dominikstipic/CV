class ApstraktniIzvor:
    def preuzmi_podatak(self):
        pass

class TipkovnickiIzvor(ApstraktniIzvor):

    def __init__(self):
        self.iter = []
    
    def preuzmi_podatak(self):
        if not self.iter:
            print("Unesi podatke")
            xs = input()
            self.iter = iter(xs.split(" "))
        try:
            return float(next(self.iter))
        except StopIteration:
            return -1
        except ValueError:
            return 0


class DatotecniIzvor(ApstraktniIzvor):
    def __init__ (self, file):
        try:
            self.iter = iter(open(file,"r"))
        except Exception:
            print("File not found")
            exit(1)

    def preuzmi_podatak(self):
        try:
            return float(next(self.iter))
        except StopIteration:
            self.iter.close()
            return -1
