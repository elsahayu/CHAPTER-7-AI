# Exercise 2.1
def minarray(xs):
    m = xs[0]
    for x in xs:
        if x < m:
            m = x
    return m


data = [15, 3, 22, 1, 8, -5, 12]
t = minarray(data)
print("Nilai minimum:", t)