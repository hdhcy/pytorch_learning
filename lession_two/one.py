'''
Author: hdhcy
Email: 982768496@qq.com

date: 2019/8/13 10:24
'''


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_dencode(i, prediction):
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][prediction]


def helper(i):
    print(fizz_buzz_dencode(i, fizz_buzz_encode(i)))

for i in range(1,15):
    helper(i)

