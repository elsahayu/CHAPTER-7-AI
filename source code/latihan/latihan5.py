# Exercise 2.5
import math

s = input("Input a list of float numbers separated by space: ")
numbers = list(map(float, s.split()))

sin_values = []
for num in numbers:
    sin_values.append(math.sin(num))

print("Original numbers :", numbers)
print("Sine values      :", sin_values)