from  SlijedBrojeva import SlijedBrojeva
import Akcije 
import Izvori




if __name__ == '__main__':
    import os
    print(os.getcwd())

    izvorDatoteka = Izvori.DatotecniIzvor("/home/doms/Desktop/Code/OO/zad6/res/data.txt")
    izvorTipkovnica = Izvori.TipkovnickiIzvor()
    
    akcijaStatistika = Akcije.Stats()
    akcijaLogger = Akcije.Logger("/home/doms/Desktop/Code/OO/zad6/res/log.txt")

    sb = SlijedBrojeva(izvorTipkovnica)
    sb.kreni([akcijaLogger, akcijaStatistika])
