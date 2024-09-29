import numpy as np

#Создаёт матрицу G 7 на 4
def create_G_Private(n, k):
    # Инициализируем матрицу нулями
    I = np.eye(k, k, 0, int)
    X = np.zeros((k, n - k), int)
    X[0] = np.array([1, 1, 0], int)
    X[1] = np.array([1, 0, 1], int)
    X[2] = np.array([0, 1, 1], int)
    X[3] = np.array([1, 1, 1], int)
    G = np.hstack((I,X))

    return G

#Создаёт матрицу G 15 на 4
def create_G_Public(n, k, d):
    I = np.eye(k, k, 0, int)
    X = np.zeros((k, n - k), int)
    X[0] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1], int)
    X[1] = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], int)
    X[2] = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], int)
    X[3] = np.array([1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1], int)
    G = np.hstack((I, X))
    return G

#создаёт матрицу H
def create_H_M(G):
    rows, cols = G.shape
    I = np.eye(n - k, n - k, 0, int)
    for i in range(rows):
        G = np.delete(G, 0, 1)
    H = np.concatenate([G, I], axis = 0)
    return H

#создаёт матрицу синдромов для единичной ошибки
def create_S_OE(U,G,H):
    I = np.eye(n, n, 0, int)
    U = np.mod(np.dot(U, G), 2)
    S = np.zeros((H.shape), int)
    for j in range(n):
        I[j] = U ^ I[j]
        S[j] = np.mod(np.dot(I[j], H), 2)
    return S

#создаёт матрицу синдромов для двойной ошибки
def create_S_TE(U, G, H):
    I = np.zeros((1, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            e = np.zeros((1, n), int)
            e[0][i] = 1
            e[0][j] = 1
            I = np.concatenate([I, e], axis = 0)
    I = np.delete(I, 0, axis=0)
    U = np.mod(np.dot(U,G),2)
    S = np.zeros((len(I), len(H[0])), int)
    for j in range(len(I)):
        I[j] = U ^ I[j]
        S[j] = np.mod(np.dot(I[j], H), 2)
    return S

#исправляет 1 на 0 и 0 на 1 по индексу
def Fix(e1, i):
    if (e1[i] == 0):
        e1[i] = 1
    else:
        e1[i] = 0
    return e1

#нахождение индекса однократной ошибки и исправление её (для первой части)
def fix_one_part1(U, G, H):
    U = np.mod(np.dot(U, G), 2)
    e1 = np.array([ 0, 0, 0, 1, 0, 0, 0])
    e1 = e1 ^ U
    s = np.mod(np.dot(e1, H), 2)
    i = 0
    while not(np.array_equal(H[i], s)):
        i=i+1
    e1 = Fix(e1, i)
    return np.array_equal(U, e1)

#нахождение индекса двукратной ошибки и исправление её (для первой части)
def fix_two_part1(U, G, H):
    U = np.mod(np.dot(U, G), 2)
    e1 = np.array([ 0, 0, 0, 0, 0, 1, 1])
    e1 = e1 ^ U
    s = np.mod(np.dot(e1, H), 2)
    i = -1
    j = 1
    Y = False
    while not (i == (len(H) - 1) or Y):
        Y = False
        i = i + 1
        j = i + 1
        while not (j == (len(H)) or Y):
            if(np.array_equal(H[i] ^ H[j], s)):
                Y = True
            else:
                j = j + 1
    if(Y):
        e1 = Fix(e1, i)
        e1 = Fix(e1, j)
    return np.array_equal(U, e1)


#нахождение индекса однократной ошибки и исправление её (для второй части)
def fix_one_part2(U, G, H):
    U = np.mod(np.dot(U, G), 2)
    e1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    e1 = e1 ^ U
    s = np.mod(np.dot(e1, H), 2)
    i = 0
    while not(np.array_equal(H[i], s)):
        i = i + 1
    e1 = Fix(e1, i)
    return np.array_equal(U, e1)

#нахождение индекса двукратной ошибки и исправление её (для второй части)
def fix_two_part2(U, G, H):
    U = np.mod(np.dot(U, G), 2)
    e1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    e1 = e1 ^ U
    s = np.mod(np.dot(e1, H), 2)
    i = -1
    j = 1
    Y = False
    while not (i == (len(H) - 1) or Y):
        Y = False
        i = i + 1
        j = i + 1
        while not (j == (len(H)) or Y):
            if(np.array_equal(H[i]^H[j], s)):
                Y=True
            else:
                j = j + 1
    if(Y):
        e1 = Fix(e1, i)
        e1 = Fix(e1, j)
    return np.array_equal(U, e1)

#нахождение индекса трёхкратной ошибки и исправление её (для второй части)
def fix_three_part2(U, G, H):
    U = np.mod(np.dot(U, G), 2)
    e1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    e1 = e1 ^ U
    s = np.mod(np.dot(e1, H), 2)
    i = -1
    j = 0
    l = 0
    Y = False
    while not (i == (len(H) - 2) or Y):
        Y = False
        i = i + 1
        j = i + 1
        l = j + 1
        while not (j == (len(H) - 1) or Y):
            j = j + 1
            l = j + 1
            while not (l == (len(H)) or Y):
                if(np.array_equal(H[i]^H[j]^H[l], s)):
                    Y = True
                else:
                    l = l + 1
    if(Y):
        e1 = Fix(e1, i)
        e1 = Fix(e1, j)
        e1 = Fix(e1, l)
    return np.array_equal(U, e1)

print("Часть 1")
n = 7
k = 4
u = np.array([1, 0, 0, 1])
G_part1 = create_G_Private(n, k)
print("\n2.1 Сформировать порождающую матрицу линейного кода (7, 4, 3).\n", "G =\n", G_part1)
H_part1 = create_H_M(G_part1)
print("\n2.2 Сформировать проверочную матрицу на основе порождающей.\n", "H =\n", H_part1)
Sind_part1 = create_S_OE(u, G_part1, H_part1)
print("\n2.3 Сформировать таблицу синдромов для всех однократных ошибок.\n", Sind_part1)
fix_o_part1 = fix_one_part1(u, G_part1, H_part1)
print("\n2.4 Внести однократную ошибку в сформированное слово. Убедиться в правильности полученного слова.\n", fix_o_part1)
fix_tw_part1 = fix_two_part1(u, G_part1, H_part1)
print("\n2.5 Внести двукратную ошибку в сформированное слово. Убедиться, что полученное слово отличается от отправленного.\n", fix_tw_part1)

print("\nЧасть 2")
n = 15
k = 4
d = 5
if(n - k >= 5 and k > 1):
    G_part2 = create_G_Public(n, k, d)
    print("\n2.6 Сформировать порождающую матрицу линейного кода (n, k, 5).\n", G_part2)
    H_part2 = create_H_M(G_part2)
    print("\n2.7 Сформировать проверочную матрицу на основе порождающей.\n", H_part2)
    u =[0, 0, 1, 0]
    Sind_One_part2 = create_S_OE(u, G_part2, H_part2)
    print("\n2.8 Сформировать таблицу синдромов для всех однократных ошибок.\n", Sind_One_part2)
    Sind_Two_part2 = create_S_TE(u, G_part2, H_part2)
    print("\n2.8 Сформировать таблицу синдромов для всех двукратных ошибок.\n", Sind_Two_part2)
    fix_o_part2 = fix_one_part2(u,G_part2,H_part2)
    print("\n2.9 Внести однократную ошибку в сформированное слово. Убедиться в правильности полученного слова.\n", fix_o_part2)
    fix_tw_part2 = fix_two_part2(u, G_part2, H_part2)
    print("\n2.10 Внести двукратную ошибку в сформированное слово. Убедиться в правильности полученного слова.\n", fix_tw_part2)
    fix_th_part2 = fix_three_part2(u, G_part2, H_part2)
    print("\n2.11 Внести трёхкратную ошибку в сформированное слово. Убедиться, что полученное слово отличается от отправленного.\n", fix_th_part2)





