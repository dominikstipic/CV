import math

# polymorph function
def mymax(iterable,key=lambda x : x ):
    max_y = max_key = []

    for x in iterable:
        if not max_y or key(x) > key(max_key):
            max_y = key(x)
            max_key = x
    return max_y,max_key

if __name__ == "__main__":
    maxint,_ = mymax([1, 3, 5, 7, 4, 6, 9, 2, 0])
    print("max int: {}".format(maxint))

    maxchar,_ = mymax("Suncana strana ulice")
    print("max char: {}".format(maxchar))

    s = [
    "Gle", "malu", "vocku", "poslije", "kise",
    "Puna", "je", "kapi", "pa", "ih", "njise"]
    maxstring,_ = mymax(s,lambda x:x[::-1])
    print("max str: {}".format(maxstring))

    D = {"burek":8, "buhtla":5}
    max_value,max_key = mymax(D, lambda x : D.get(x))
    print("max dict value: {}".format(max_value))
    print("max dict key: {}".format(max_key))

    persons = [("Ivan","Ivic"),("Maro","Marulic"),("Danko","Danic"),("zdravko","zdravic"),("ante", "zdravic")]
    ime,t = mymax(persons, lambda x : x[1])
    print("ime: {}".format(ime))
    print("sve: {}".format(t))





