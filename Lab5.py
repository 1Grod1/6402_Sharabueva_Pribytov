import numpy as np
import math
from itertools import combinations, product

# Формируем вектор V_I на основе множества I
def Vec_I(subset, col):
    if len(subset) == 0:
        return np.ones(2 ** col, int)
    else:
        v = []
        for bin_v in list(product([0, 1], repeat=col)):
            f = np.prod([(bin_v[i] + 1) % 2 for i in subset])
            v.append(f)
        return v

# Формирование порождающей матрицы G для кода Рида-Маллера
def crate_RM_G(r, m):
    matr_size = sum(math.comb(m, i) for i in range(r + 1))
    matr = np.zeros((matr_size, 2 ** m), dtype=int)
    indices = range(m)
    combs = [subset for subset_size in range(r + 1) for subset in combinations(indices, subset_size)]
    i = 0
    for subset in combs:
        matr[i] = Vec_I(subset, m)
        i += 1
    return matr


def Vec_H(I, m):
    return [word for word in list(product([0, 1], repeat=m)) if np.prod([(word[idx] + 1) % 2 for idx in I]) == 1]

def Vec_I_t(I, m, t):
    if not I:
        return np.ones(2 ** m, dtype=int)
    return [np.prod([(word[j] + t[j] + 1) % 2 for j in I]) for word in list(product([0, 1], repeat=m))]


# Сортировка для декодирования
def Sort_major(m, r):
    ind = range(m)
    comb_list = list(combinations(ind, r))

    if comb_list:
        comb_list.sort(key=lambda x: len(x))
    result = np.array(comb_list, dtype=int)
    return result


# Мажоритарное декодирование
def major_algorithm(w, r, m, size):
    i = r
    w_r = w.copy()
    Mi = np.zeros(size, dtype=int)
    max_weight = 2 ** (m - r - 1) - 1
    idx = 0

    while True:
        for J in Sort_major(m, i):
            max_count_z_o = 2 ** (m - i - 1)
            zeros_count = 0
            ones_count = 0
            for t in Vec_H(J, m):
                dop = [i for i in range(m) if i not in J]
                V = Vec_I_t(dop, m, t)
                c = np.dot(w_r, V) % 2

                if c == 0:
                    zeros_count += 1
                else:  # c == 1
                    ones_count += 1

            if zeros_count > max_weight and ones_count > max_weight:
                return

            if zeros_count > max_count_z_o:
                Mi[idx] = 0
                idx += 1
            if ones_count > max_count_z_o:
                Mi[idx] = 1
                idx += 1
                V = Vec_I(J, m)
                w_r = (w_r + V) % 2

        if i > 0:
            if len(w_r) < max_weight:
                for J in Sort_major(m, r + 1):
                    Mi[idx] = 0
                    idx += 1
                break
            i -= 1
        else:
            break

    reversed(Mi)
    return Mi


# Генерация слова с указанным количеством ошибок
def mistakes(G, error_count,u):
    u = u.dot(G) % 2
    mistake_pos = np.random.choice(len(u), size=error_count, replace=False)
    u[mistake_pos] = (u[mistake_pos] + 1) % 2
    return u


# Генерация порождающей матрицы кода Рида-Маллера
G_matr = crate_RM_G(2, 4)
print("Порождающая матрица кода Рида-Маллера:\n", G_matr,"\n\n||||||||||||||||||||||||||||||||||||||||||||")

print("Для однократной ошибки\n")
# Эксперимент для однократной ошибки
u = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
print("Исходное сообщение:", u)
Err = mistakes(G_matr, 1, u)
print("Слово с ошибкой:", Err)

Decoded_word = major_algorithm(Err, 2, 4, len(G_matr))
if Decoded_word is None:
    print("\nERROR")
else:
    V2 = Decoded_word.dot(G_matr) % 2
    print("Результат u*G:", V2)
    print("Исправленное слово:", Decoded_word,"\n||||||||||||||||||||||||||||||||||||||||||||\n")


print("Для двукратной ошибки\n")
# Эксперимент для двукратной ошибки
u = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])

print("Исходное сообщение:", u)
Err = mistakes(G_matr, 2, u)
print("Слово с ошибкой:", Err)

Decoded_word = major_algorithm(Err, 2, 4, len(G_matr))
if Decoded_word is None:
   print("\nERROR\n ошибка не исправлена")
else:
   V2 = Decoded_word.dot(G_matr) % 2
   print("Результат u*G:", V2)
   print("Исправленное слово:", Decoded_word)
