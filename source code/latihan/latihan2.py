# Exercise 2.2
def sortarray(xs):
    arr = list(xs)  # Membuat salinan list
    n = len(arr)
    # Implementasi bubble sort
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


data = [5, 2, 9, 1, 5, 6]
sorted_data = sortarray(data)
print("Array setelah diurutkan:", sorted_data)