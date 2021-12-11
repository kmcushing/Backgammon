import math


def calc(a, b):
    return a**2-(19*(b**2))


def find_ans():
    a1 = 3
    b1 = 1
    a2 = 6
    b2 = 1
    i = 2
    while not calc(a1+a2, b1+b2) == 1 or not abs((a1+a2)/(b1+b2)-math.sqrt(19)) < (1/2/((b1+b2)**2)):
        if i > 15:
            break
        print((a1+a2)/(b1+b2))
        print("val")
        print(calc(a1+a2, b1+b2))
        print("error")
        print(abs((a1+a2)/(b1+b2)-math.sqrt(19)) < (1/2/((b1+b2)**2)))
        print((1/2/((b1+b2)**2)))
        print(abs((a1+a2)/(b1+b2)-math.sqrt(19)))
        if (a1+a2)/(b1+b2) < math.sqrt(19):
            a1 = a1+a2
            b1 = b1+b2
            print(f"lower {a1}/{b1}")
        else:
            a2 = a2+a1
            b2 = b1+b2
            print(f"upper {a2}/{b2}")
        i += 1
    print(f"found {a1+a2}/{b1+b2}")


find_ans()
