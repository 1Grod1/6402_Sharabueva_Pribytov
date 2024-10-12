import numpy as np
import itertools

#функция формирования проверочной матрицы для кода Хэмминга
def generate_H(m, n):
    combinations = list(itertools.product([0, 1], repeat=n))

    H = np.zeros((m, n)).astype(int)
    H[-n:,:] = np.eye(n)
    i = 0
    for combination in combinations:
        if sum(combination) > 1:
            H[i, :] = combination
            i += 1

    return H

#функция формирования пораждающей матрицы для кода Хэмминга
def generate_G(n,k , H):
    G = np.zeros((k, n), dtype=int)
    # В первых k столбцах - единичная матрица
    G[:, 0:k] = np.eye(k, dtype=int)
    G[:,k:] =H[0: k, :].copy()

    return G


#функция формирования проверочной матрицы для расширенного кода Хэмминга
def generate_H_Rush(n, m):
    combinations = list(itertools.product([0, 1], repeat=m))

    H = np.zeros((n, m)).astype(int)
    H[-m:,:] = np.eye(m)
    i = 0
    for combination in combinations:
        if sum(combination) > 1:
            H[i, :] = combination
            i += 1
    H=np.concatenate([H, [np.array([0]*m)]], axis=0)
    v_vert_ones = np.ones((n+1, 1)).astype(int)
    H = np.hstack((H, v_vert_ones))

    return H

#функция формирования порождающей матрицы для расширенного кода Хэмминга
def generate_G_Rush(n,k , H):
    G = np.zeros((k, n), dtype=int)
    # В первых k столбцах - единичная матрица
    G[:, 0:k] = np.eye(k, dtype=int)
    G[:,k:] =H[0: k,:len(H[0])-1].copy()
    v_vert_ones = np.ones((k, 1)).astype(int)
    G = np.hstack((G, v_vert_ones))
    for row in G:
        if (sum(row)%2)!=0:
            row[n]=0
    return G

#функция формирования таблицы синдромов для кода Хэмминга
def syndrome_table(G, H):
    U=np.array([0]*len(G))
    I = np.eye(len(H), len(H), 0, int)
    U = np.mod(np.dot(U, G), 2)
    S = np.zeros((H.shape), int)
    for j in range(len(H)):
        I[j] = U ^ I[j]
        S[j] = np.mod(np.dot(I[j], H), 2)
    return S

#Функция обнаружения и исправления однократной ошибки
def fix_one(G,S_matr):
    U = np.array([0] * len(G))
    U = np.mod(np.dot(U, G), 2)
    e1=U.copy()
    e1[0] = (e1[0] + 1) % 2
    s=np.mod(np.dot(e1, S_matr), 2)
    i=0
    while not(np.array_equal(S_matr[i],s)):
        i=i+1
    e1[i]=(e1[i] + 1) % 2
    return np.array_equal(U,e1)

#Функция обнаружения и попытки исправления двухкратной ошибки
def fix_two(G,S_matr):
    U = np.array([0] * len(G))
    U = np.mod(np.dot(U, G), 2)
    e1=U.copy()
    e1[0] = (e1[0] + 1) % 2
    e1[1] = (e1[1] + 1) % 2
    s=np.mod(np.dot(e1, S_matr), 2)
    if (np.array_equal(s, np.zeros(len(s), dtype=int))):
        return ("ошибка не найдена")
    else:
        i = 0
        for j in range(len(S_matr)):
            if (np.array_equal(S_matr[j], s)):
                i = j
        e1[i] = (e1[i] + 1) % 2
        return np.array_equal(U, e1), "ошибка обнаружена"

#Функция обнаружения и попытки исправления трёхкратной ошибки
def fix_three(G,S_matr):
    U = np.array([0] * len(G))
    U = np.mod(np.dot(U, G), 2)
    e1=U.copy()
    e1[0] = (e1[0] + 1) % 2
    e1[1] = (e1[1] + 1) % 2
    e1[4] = (e1[4] + 1) % 2
    s=np.mod(np.dot(e1, S_matr), 2)

    if (np.array_equal(s, np.zeros(len(s), dtype=int))):
        return("ошибка не найдена")
    else:
        i = 0
        for j in range(len(S_matr)):
            if (np.array_equal(S_matr[j], s)):
                i = j
        e1[i] = (e1[i] + 1) % 2
        return np.array_equal(U, e1), "ошибка обнаружена"

#Функция обнаружения и попытки исправления четырёхкратной ошибки
def fix_four(G,S_matr):
    U = np.array([0] * len(G))
    U = np.mod(np.dot(U, G), 2)
    e1=U.copy()
    e1[0] = (e1[0] + 1) % 2
    e1[1] = (e1[1] + 1) % 2
    e1[4] = (e1[4] + 1) % 2
    e1[6] = (e1[6] + 1) % 2
    s=np.mod(np.dot(e1, S_matr), 2)

    if (np.array_equal(s, np.zeros(len(s), dtype=int))):
        return("ошибка не найдена")
    else:
        i = 0
        for j in range(len(S_matr)):
            if (np.array_equal(S_matr[j], s)):
                i = j
        e1[i] = (e1[i] + 1) % 2
        return np.array_equal(U, e1), "ошибка обнаружена"


r=2
n = 2 ** r - 1  # Длина закодированного слова
k = 2 ** r - r - 1  # Длина сообщения
m = n - k  # Количество проверочных битов

H_matr = generate_H(n, r)
G_matr = generate_G(n,k,H_matr)
Syndr_matr = syndrome_table(G_matr,H_matr)
print("Результаты кода Хэмминга для r = 2:")
print("Матрица H")
print(H_matr)
print("Матрица G")
print(G_matr)
print("таблица синдромов")
print(Syndr_matr)
print("для однократной ошибки",fix_one(G_matr,Syndr_matr))
print("для двукратной ошибки",fix_two(G_matr,Syndr_matr),'\n')


HR_matr = generate_H_Rush(n, r)
GR_matr = generate_G_Rush(n,k,HR_matr)
Syndr_R_matr = syndrome_table(GR_matr,HR_matr)
print("Результаты расширенного кода Хэмминга для r = 2:")
print("Матрица H")
print(HR_matr)
print("Матрица G")
print(GR_matr)
print("таблица синдромов")
print(Syndr_R_matr)
print("для однократной ошибки",fix_one(GR_matr,Syndr_R_matr))
print("для двукратной ошибки",fix_two(GR_matr,Syndr_R_matr),'\n\n')

#
r2=3
n2 = 2 ** r2 - 1  # Длина закодированного слова
k2 = 2 ** r2 - r2 - 1  # Длина сообщения
m2 = n2 - k2  # Количество проверочных битов

H_matr_2 = generate_H(n2, r2)
G_matr_2 = generate_G(n2,k2,H_matr_2)
Syndr_matr_2 = syndrome_table(G_matr_2,H_matr_2)
print("Результаты кода Хэмминга для r = 3:")
print("Матрица H")
print(H_matr_2)
print("Матрица G")
print(G_matr_2)
print("таблица синдромов")
print(Syndr_matr_2)
print("для однократной ошибки",fix_one(G_matr_2,Syndr_matr_2))
print("для двукратной ошибки",fix_two(G_matr_2,Syndr_matr_2))
print("для трёхкратной ошибки",fix_three(G_matr_2,Syndr_matr_2),'\n')


HR_matr_2 = generate_H_Rush(n2, r2)
GR_matr_2 = generate_G_Rush(n2,k2,HR_matr_2)
Syndr_R_matr_2 = syndrome_table(GR_matr_2,HR_matr_2)
print("Результаты расширенного кода Хэмминга для r = 3:")
print("Матрица H")
print(HR_matr_2)
print("Матрица G")
print(GR_matr_2)
print("таблица синдромов")
print(Syndr_R_matr_2)
print("для однократной ошибки",fix_one(GR_matr_2,Syndr_R_matr_2))
print("для двукратной ошибки",fix_two(GR_matr_2,Syndr_R_matr_2))
print("для трёхкратной ошибки",fix_three(GR_matr_2,Syndr_R_matr_2))
print("для четырехкратной ошибки",fix_four(GR_matr_2,Syndr_R_matr_2),'\n\n')



r3=4
n3 = 2 ** r3 - 1  # Длина закодированного слова
k3 = 2 ** r3 - r3 - 1  # Длина сообщения
m3 = n3 - k3  # Количество проверочных битов

H_matr_3 = generate_H(n3, r3)
G_matr_3 = generate_G(n3,k3,H_matr_3)
Syndr_matr_3 = syndrome_table(G_matr_3,H_matr_3)
print("Результаты кода Хэмминга для r = 4:")
print("Матрица H")
print(H_matr_3)
print("Матрица G")
print(G_matr_3)
print("таблица синдромов")
print(Syndr_matr_3)
print("для однократной ошибки",fix_one(G_matr_3,Syndr_matr_3))
print("для двукратной ошибки",fix_two(G_matr_3,Syndr_matr_3))
print("для трёхкратной ошибки",fix_three(G_matr_3,Syndr_matr_3),'\n')


HR_matr_3 = generate_H_Rush(n3, r3)
GR_matr_3 = generate_G_Rush(n3,k3,HR_matr_3)
Syndr_R_matr_3 = syndrome_table(GR_matr_3,HR_matr_3)
print("Результаты расширенного кода Хэмминга для r = 4:")
print("Матрица H")
print(HR_matr_3)
print("Матрица G")
print(GR_matr_3)
print("таблица синдромов")
print(Syndr_R_matr_3)
print("для однократной ошибки",fix_one(GR_matr_3,Syndr_R_matr_3))
print("для двукратной ошибки",fix_two(GR_matr_3,Syndr_R_matr_3))
print("для трёхкратной ошибки",fix_three(GR_matr_3,Syndr_R_matr_3))
print("для четырехкратной ошибки",fix_four(GR_matr_3,Syndr_R_matr_3))