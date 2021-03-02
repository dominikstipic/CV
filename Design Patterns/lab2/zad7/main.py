from Sheet import Sheet


if __name__ == '__main__':
    s = Sheet(5,5)
    
    while(True):
        s.printSheet()
        user_input = input()
        user_input = user_input.strip().split(" ")
        try:
            s.set(user_input[0],user_input[1])
        except AssertionError as e:
            print(e)
        except IndexError as i:
            print(i)