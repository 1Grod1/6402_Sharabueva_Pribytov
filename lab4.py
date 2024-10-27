import numpy as np

#Функция создания пораждающей и проверочной матриц расширенного кода Голея
def create_G_H():
    G = np.hstack((np.eye(12, 12, dtype=int), B))
    H = np.concatenate((np.eye(12, 12, dtype=int),B), axis=0)
    return G,H


#Функция обнаружения и исправления n-кратной ошибки с помощью расширенного кода Голея
def fix_error_G(G,H_matr,n):
    U = np.array([0] * len(G))
    w = np.mod(np.dot(U, G), 2)
    e1=w.copy()
    for i in range(n):
        e1[i] = (e1[i] + 1) % 2
    s=np.mod(np.dot(e1, H_matr), 2)
    u1=None
    if (sum(s)<=3):
        u1 = np.hstack((s, np.zeros(len(s), dtype=int)))
    for i in range(len(B)):
        if(sum(s^B[i])<=2):
            ei=np.zeros(len(s))
            ei[i]=1
            u1=np.hstack(s^B[i],ei)
    if u1 is None:
        sec_s = np.mod(np.dot(s, B), 2)
        if (sum(sec_s) <= 3):
            u1 = np.hstack(( np.zeros(len(s), dtype=int),sec_s))
        for j in range(len(B)):
            if (sum(sec_s ^ B[j]) <= 2):
                ei = np.zeros(len(s))
                ei[j] = 1
                u1 = np.hstack(ei,sec_s ^ B[j])
    if u1 is not None:
        u1=u1^e1
    if np.array_equal(u1,w):
        message="ошибка обнаружена и исправлена"
    else:
        message = "ошибка обнаружена, но не исправлена"
    return message


#Функция создания пораждающей матрицы кода Рида-Маллера
def crate_RM_G(r, m):
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    if r == m:
        G_mm = crate_RM_G(m - 1, m)
        row = np.zeros((1, 2 ** m), dtype=int)
        row[0, -1] = 1
        return np.vstack([G_mm, row])

    # Рекурсивный случай: [[G(r, m-1), G(r, m-1)],[0, G(r-1, m-1)]]
    G_rm1 = crate_RM_G(r, m - 1)
    G_r1_m1 = crate_RM_G(r - 1, m - 1)
    # Верхняя часть: G(r, m-1) дублируется
    up = np.hstack([G_rm1, G_rm1])
    # Нижняя часть: нули слева и G(r-1, m-1) справа
    dow = np.hstack([np.zeros((G_r1_m1.shape[0], G_r1_m1.shape[1]), dtype=int), G_r1_m1])
    G = np.vstack([up, dow])
    return G


#Функция создания проверочной матрицы кода Рида-Маллера
def H_im_matr(i,m):
    H=np.array([[1,1],[1,-1]],dtype=int)
    I1=np.eye(2**(m-i), dtype=int)
    I2=np.eye(2**(i-1), dtype=int)
    H_m = np.kron(I1,H)
    H_m =np.kron(H_m,I2)
    return H_m

#Функция заменяющая 0 на -1
def w_w1(w):
    w1=np.array([1]*len(w))
    for j in range(len(w)):
        if(w[j]==0):
            w1[j]=-1
    return w1

#Функция обнаружения и исправления n-кратной ошибки с помощью кода Рида-Маллера
def fix_error_RM(G,m,n):
    U = np.array([0] * len(G)) # исходное сообщение
    w = np.mod(np.dot(U, G), 2) #получаем U*G
    e1 = w.copy()
    for i in range(n):# делаем ошибки
        e1[i] = (e1[i] + 1) % 2
    #замена 0 на -1
    wm=np.dot(w_w1(e1), H_im_matr(1,m))
    #вычисляем wm и индекс максимального абсолютного значения в wm
    for j in range(2, m + 1):
        wm = np.dot(wm, H_im_matr(j, m))
    lead = np.argmax(abs(wm))

    v=np.array([0]*m)
    V_bin=bin(lead)[2:]
    V_rev=list(V_bin)
    V_rev.reverse()
    for j in range(len(V_rev)):
        v[j]=int(V_rev[j])
    if(wm[lead]>0):
        u = np.append(np.array([1],dtype=int), v)
    else:
        u = np.append(np.array([0],dtype=int), v)

    if np.array_equal(u,U):
        message = "ошибка обнаружена и исправлена"
    else:
        message = "ошибка обнаружена, но не исправлена"
    return message


B = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
              [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
              [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
              [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
              [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
              [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
              [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
              [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
              [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])


print("Часть 1")
G_matr,H_matr = create_G_H()
print("Матрица G расширенного кода Голея\n",G_matr)
print("Матрица H расширенного кода Голея\n",H_matr)
print("Исправление ошибок с помощью расширенного кода Голея")
print("для ошибки кратности ",1,": ",fix_error_G(G_matr,H_matr,1))
print("для ошибки кратности ",2,": ",fix_error_G(G_matr,H_matr,2))
print("для ошибки кратности ",3,": ",fix_error_G(G_matr,H_matr,3))
print("для ошибки кратности ",4,": ",fix_error_G(G_matr,H_matr,4),'\n\n')


print("Часть 2")
r,m=1,3
RM_G_matr=crate_RM_G(r,m)
print("Матрица G кода Рида-Маллера r=1,m=3\n",RM_G_matr)
print("Исправление ошибок с помощью кода Рида-Маллера при r = 1, m = 3")
print("для ошибки кратности ",1,": ",fix_error_RM(RM_G_matr,m,1))
print("для ошибки кратности ",2,": ",fix_error_RM(RM_G_matr,m,2),'\n\n')

r,m=1,4
RM_G_matr=crate_RM_G(r,m)
print("Матрица G кода Рида-Маллера, r = 1,m = 4\n",RM_G_matr)
print("Исправление ошибок с помощью кода Рида-Маллера при r = 1, m = 4")
print("для ошибки кратности ",1,": ",fix_error_RM(RM_G_matr,m,1))
print("для ошибки кратности ",2,": ",fix_error_RM(RM_G_matr,m,2))
print("для ошибки кратности ",3,": ",fix_error_RM(RM_G_matr,m,3))
print("для ошибки кратности ",4,": ",fix_error_RM(RM_G_matr,m,4))