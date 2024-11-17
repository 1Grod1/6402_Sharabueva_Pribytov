import numpy as np
import random

#Функция для разделения многочлена на многочлен с остатком
def remainder(dividend, divisor):
    remainder = np.array(dividend)
    len_divisor = len(divisor)

    while len(remainder) >= len_divisor:
        shift = len(remainder) - len_divisor
        remainder[shift:shift + len_divisor] = [r ^ d for r, d in zip(remainder[shift:shift + len_divisor], divisor)]

        # Удаляем последние нули
        while len(remainder) > 0 and remainder[-1] == 0:
            remainder = remainder[:-1]  # remainder.pop()

    return np.array(remainder)

#Функция произведения многочленов
def multiply(A, B):
    result = np.zeros(len(A) + len(B) - 1, dtype=int)

    for i in range(len(B)):
        if B[i] == 1:  # если коэффициент в B ненулевой
            result[i:i + len(A)] ^= A.astype(int)

    return result

#функция для добавления n-кратной ошибки и попытка исправить её
def Err_and_corr(a, g, n):
    v = multiply(a, g)
    print("Отправленное слово:        ", v)

    w = v.copy()
    error = np.zeros(len(w), dtype=int)

    # Генерация ошибок в зависимости от n
    if n == 1:
        error[random.randint(0, len(w) - 1)] = 1
    elif n == 2:
        i1 = random.randint(0, len(w) - 2)
        i2 = i1 + random.choice([1, 2])
        if (i2 >= len(v)):
            i2 = i2 % len(v)
        error[i1] = error[i2] = 1
    else:
        error_ind = random.sample(range(len(w)), n)
        error[error_ind] = (error[error_ind] + 1) % 2

    w = (w + error) % 2
    print("Слово с", n, "-кратной ошибкой:", w)

    s = remainder(w, g)

    # Определяем шаблоны ошибок
    error_templates = [[1]] if n == 1 else [[1, 1, 1], [1, 0, 1], [1, 1], [1]]

    idx = 0
    found = False

    # Поиск соответствия с шаблонами ошибок
    while not found:
        if (any(np.array_equal(s, template) for template in error_templates)) or (idx > len(w) - 1):
            found = True
        else:
            s = remainder(multiply(s, np.array([0, 1])), g)
            idx += 1

    temp = np.zeros(len(w), dtype=int)
    if idx == 0:
        temp[idx] = 1
    else:
        temp[len(temp) - idx] = 1

    e = multiply(s, temp)[:len(w)]
    message = (w + e) % 2
    print("Исправленное слово:        ", message)

    # Проверка на корректность исправления
    if np.array_equal(v, message):
        print("True")
    else:
        print("False")


a = np.array([1, 0, 0, 1])
g = np.array([1, 0, 1, 1])

print("Порождающий полином:    ", g)
print("Входное слово:      ", a)

print("\n\nИсправление для однократной ошибки")
Err_and_corr(a, g, 1)
print("\n\nИсправление для двухкратной ошибки")
Err_and_corr(a, g, 2)
print("\n\nИсправление для трехкратной ошибки")
Err_and_corr(a, g, 3)

a = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1])
g = np.array([1, 0, 0, 1, 1, 1, 1])

print("\n\nПорождающий полином:    ", g)
print("Входное слово:      ", a)

print("\n\nИсправление для однократной ошибки")
Err_and_corr(a, g, 1)
print("\n\nИсправление для двухкратной ошибки")
Err_and_corr(a, g, 2)
print("\n\nИсправление для трехкратной ошибки")
Err_and_corr(a, g, 3)
print("\n\nИсправление для четырехкратной ошибки")
Err_and_corr(a, g, 4)