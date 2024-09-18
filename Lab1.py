import numpy as np
import itertools
import math


def REF(A):
    """1.1 Реализовать функцию REF(), приводящую матрицу к
        ступенчатому виду"""
    """1.3.1 На основе входной матрицы сформировать порождающую
матрицу в ступенчатом виде."""
    rows, cols = A.shape
    k=0
    
    for r in range(rows):
        # Находим индекс максимального элемента в текущем столбце
        max_row = np.argmax(A[r:rows, r])
        
        # Меняем текущую строку с строкой с максимальным элементом
        A[[r, max_row+r]] = A[[max_row+r, r]]
        #Находим индекс ведущего элемента в текущей строке
        lead = np.argmax(A[r])
        # Приведение к нулю элементов ниже текущего ведущего элемента
        for i in range(r + 1, rows):
            if A[i, lead] != 0:
                A[i] = A[i]^A[r]
    return A


def RREF(A):
    """1.2Реализовать функцию REF(), приводящую матрицу к
        ступенчатому виду"""
    rows, cols = A.shape
    for r in range(rows):
        # Находим индекс максимального элемента в текущем столбце
        max_row = np.argmax(A[r:rows, r]) + r
        
        # Меняем текущую строку с строкой с максимальным элементом
        A[[r, max_row]] = A[[max_row, r]]

        #Находим индекс ведущего элемента в текущей строке
        lead = np.argmax(A[r])
        # Приведение к нулю элементов ниже текущего ведущего элемента
        for i in range(r + 1, rows):
            if A[i, lead] != 0:
                A[i] = A[i]^A[r]
                
        #Приведение к нулю элементов выше текущего элемента
        for j in range(r,0, -1):
            if A[j-1,lead]!=0:
                A[j-1]=A[j-1]^A[r]

    #Избавляемся от нулевых строк
    B = A[~np.all(A == 0, axis = 1)]
    
    return B


"""Сформировать матрицу 𝐆∗ в приведённом ступенчатом виде на основе порождающей."""
def Gresult(A):
    rows, cols = A.shape
    leadCol = np.array([],int)
    for r in range(rows):
        # Находим индекс максимального элемента в текущем столбце
        max_row = np.argmax(A[r:rows, r]) + r
        # Меняем текущую строку с строкой с максимальным элементом
        A[[r, max_row]] = A[[max_row, r]]
        #Находим индекс ведущего элемента в текущей строке
        lead = int(np.argmax(A[r]))
        #добавляем в массив индекс столбцов с ведущими элементами
        if (lead>0 or len(leadCol)==0):
            leadCol = np.append(leadCol,[lead])
        # Приведение к нулю элементов ниже текущего ведущего элемента
        for i in range(r + 1, rows):
            if A[i, lead] != 0:
                A[i] = A[i]^A[r]
                
        #Приведение к нулю элементов выше текущего элемента
        for j in range(r,0, -1):
            if A[j-1,lead]!=0:
                A[j-1]=A[j-1]^A[r]
    #Избавляемся от нулевых строк
    B = A[~np.all(A == 0, axis = 1)]
    return B,leadCol


"""Сформировать сокращённую матрицу 𝐗, удалив ведущие столбцы матрицы 𝐆∗"""
def DelLead(A,leadCol):
    for i in reversed(leadCol):
        A = np.delete(A, i, 1)
    return A


"""Сформировать матрицу H"""
def Hmatr(A,leadCol,rows,cols):
    lenght=cols-rows
    I = np.eye(lenght,lenght,0,int)
    H = np.zeros((lenght+len(A),lenght),int)
    j=0
    k=0
    h=0
    for e in leadCol:
        if(e==h):
            H[h]=A[k]
            k=k+1
            h=h+1
        else:
            while(e!=h):
                H[h]=I[j]
                j=j+1
                h=h+1
            H[h]=A[k]
            k=k+1
            h=h+1
    while(j<lenght):
        H[h]=I[j]
        j=j+1
        h=h+1
    
    return H


"""Сложить все слова из порождающего множества, оставить неповторяющиеся"""
def SumRow(A):
    rows, cols = A.shape
    lst = list(range(rows))
    combs = []
    for i in range(2, len(lst)+1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        for num in els:
            combs = np.zeros((1,cols),int)
            for j in range(len(num)):
                combs=combs^A[num[j]]
            A=np.concatenate([A, combs], axis=0)
    for i in range(len(A)):
        for j in range(i+1,len(A)):
          if np.array_equal(A[j], A[i]):
                # Если нужно удалить дубликаты, можно использовать:
                A = np.delete(A, j)  # Удаление элемента
    return A


"""Взять все двоичные слова длины k, умножить каждое на G. """
def Ymnoz(G_matr,H_matr):
    rows,cols=G_matr.shape
    num_codes = 2 ** rows
    binary_codes = np.array([[int(bit) for bit in format(i, f'0{rows}b')] for i in range(num_codes)])
    for u in binary_codes:
        v = np.dot(u,G_matr)
        v = np.mod(v,2)
        print("строка V : ",v)
        print("строка V*H : ",np.mod(np.dot(v,H_matr),2),"\n")
    return 1


""" Вычислить кодовое расстояние получившегося кода."""
def Dist(matr):
    arr=[]
    for row in matr:
        k=0
        for i in range(len(row)):
            k=k+row[i]
        arr.append(k)
    return min(arr),math.ceil((min(arr)-1)/2)



""" Внести в кодовое слово ошибку кратности не более t, умножить
    полученное слово на H, убедиться в обнаружении ошибки"""
def IdError(H_matr,V):
    e=np.array([0,0,1,0,0,0,0,0,0,0,0],int)
    print("Заложенная ошибка ",e)
    Ve=V^e
    Res = np.mod(np.dot(Ve,H_matr),2)
    max_pos = np.argmax(Res)
    if(Res[max_pos]==1):
        print(Res," - Error\n")
    else:
        print(Res," - So good")
    return 1



""" Найти для некоторого кодового слова ошибку кратности t+1
    такую, что при умножении на H ошибка не может быть обнаружена"""
def TPlus(H_matr,V):
    e=np.array([0,0,0,0,0,0,0,0,0,0,0],int)
    er=e
    for i in range(len(e)):
        for j in range(i+1,len(e)):
            e[i]=1
            e[j]=1
            print(e)
            Ve=V^e
            Res = np.mod(np.dot(Ve,H_matr),2)
            max_pos = np.argmax(Res)
            if(Res[max_pos]==1):
                print(Res," - Error\n")
            else:
                print(Res," - So good____\n")
                er=e
            e=np.array([0,0,0,0,0,0,0,0,0,0,0],int)
    
    return er


# Пример использования
A = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int)

ref_matr = REF(A)
print("1.1 Функция REF\n",ref_matr)
rref_matrix = RREF(A)
print("\n1.2 Функция RREF\n",rref_matrix)
G_matr = REF(A)
print("\n1.3.1 Функция REF задание \n",G_matr)
G_matr = G_matr[~np.all(G_matr == 0, axis = 1)]
rows, cols = G_matr.shape
print("\n1.3.2 задание \n n = ",cols,"\nk = ",rows)
GP_matr,leadCol = Gresult(G_matr)
Xresult = DelLead(GP_matr,leadCol)
print("\n1.3.3 задание \n",Xresult,"\nLead Col = ",leadCol)
H_matr = Hmatr(Xresult,leadCol,rows, cols)
print("\n1.3.4 Функция Hmatr  задание \n",H_matr)
print(G_matr)
sum_matr = SumRow(G_matr)
D,t=Dist(G_matr)
V= np.array([1,0,1,1,1,0,1,0,0,1,0],int)
print("\n1.4.1 Функция SumRow  задание \n",sum_matr)
print("\n1.4.2 Функция Ymnoz  задание \n")
real = Ymnoz(G_matr,H_matr)
print("\n1.5 Функция  Dist задание \n Input:\n",G_matr,"\n d = ",D,"\n t = ", t)
print("\n1.5.1 Функция IdError  задание\n")
IdError(H_matr,V)
print("\n1.5.2 Функция IdError  задание \n")
Er = TPlus(H_matr,V)
print(" e2, при которой не обнаруживается ошибка = ", Er)
