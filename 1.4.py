import numpy as np
import itertools
import math

def NotPriv_Gresult(A):
    rows, cols = A.shape
    for r in range(rows):
        # Находим индекс максимального элемента в текущем столбце
        max_row = np.argmax(A[r:rows, r]) + r
        
        # Меняем текущую строку с строкой с максимальным элементом
        A[[r, max_row]] = A[[max_row, r]]

        #Находим индекс ведущего элемента в текущей строке
        lead = int(np.argmax(A[r]))
        # Приведение к нулю элементов ниже текущего ведущего элемента
        for i in range(r + 1, rows):
            if A[i, lead] != 0:
                A[i] = A[i]^A[r]
    #Избавляемся от нулевых строк
   
    B = A[~np.all(A == 0, axis = 1)]
    
    return B



def Priv_Gresult(A):
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
    #############################            
        #Приведение к нулю элементов выше текущего элемента
        for j in range(r,0, -1):
            if A[j-1,lead]!=0:
                A[j-1]=A[j-1]^A[r]
                ############################
    #Избавляемся от нулевых строк
    A = A[~np.all(A == 0, axis = 1)]
    return A,leadCol

def DelLead(A,leadCol):
    for i in reversed(leadCol):
        #print(i)
        A = np.delete(A, i, 1)
        #for j in range(len(A)):
            #print(A[j,i])
            
        #for j in range(len(A)):
             #A[j].pop(i)
    return A


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

def SumRow(A):
    rows, cols = A.shape
    lst = list(range(rows))
    combs = []
    for i in range(2, len(lst)+1):
        #combs.append(i)
        els = [list(x) for x in itertools.combinations(lst, i)]
        #print(els,'\n')
        for num in els:
            combs = np.zeros((1,cols),int)
            for j in range(len(num)):
                combs=combs^A[num[j]]
            A=np.concatenate([A, combs], axis=0)
            #print(num,"    ",combs,'\n')
        #combs.append(els)
    #print(A.shape)
    for i in range(len(A)):
        for j in range(i+1,len(A)):
          if np.array_equal(A[j], A[i]):
                # Если нужно удалить дубликаты, можно использовать:
                A = np.delete(A, j)  # Удаление элемента
    #print(A.shape)
    return A

def Ymnoz(G_matr,H_matr):
    print("\nФункция Ymnoz  задание 1.4.2\n")
    rows,cols=G_matr.shape
    num_codes = 2 ** rows
    binary_codes = np.array([[int(bit) for bit in format(i, f'0{rows}b')] for i in range(num_codes)])
    #print(binary_codes)
    for u in binary_codes:
        v = np.dot(u,G_matr)
        v = np.mod(v,2)
        print("строка V : ",v,"\n")
        print("строка V*H : ",np.mod(np.dot(v,H_matr),2),"\n")
    return 1

def Dist(matr):
    arr=[]
    for row in matr:
        k=0
        for i in range(len(row)):
            k=k+row[i]
        arr.append(k)
    return min(arr),math.ceil((min(arr)-1)/2)


def IdError(H_matr,V):
    print("\nФункция IdError  задание 1.5.1\n")
    e=np.array([0,0,1,0,0,0,0,0,0,0,0],int)
    Ve=V^e
    Res = np.mod(np.dot(Ve,H_matr),2)
    max_pos = np.argmax(Res)
    if(Res[max_pos]==1):
        print(Res," - Error\n")
    else:
        print(Res," - So good")
    return 1

def TPlus(H_matr,V):
    print("\nФункция IdError  задание 1.5.2\n")
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
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]],int)

G_matr = NotPriv_Gresult(A)
PrG_matr,leadCol = Priv_Gresult(A)
rows, cols = G_matr.shape
Xresult = DelLead(PrG_matr,leadCol)
H_matr = Hmatr(Xresult,leadCol,rows, cols)
sum_matr = SumRow(G_matr)
D,t=Dist(G_matr)
V= np.array([1,0,1,1,1,0,1,0,0,1,0],int)
print("\nФункция Gresult задание 1.3.1\n",G_matr)
print("\nФункция Hmatr  задание 1.3.4\n",H_matr)
print("\nФункция SumRow  задание 1.4.1\n",sum_matr)
real = Ymnoz(G_matr,H_matr)
print("\nФункция  Dist задание 1.5\n Input:\n",G_matr,"\n d = ",D,"\n t = ", t)
IdError(H_matr,V)
Er = TPlus(H_matr,V)
print(" e2, при которой не обнаруживается ошибка = ", Er)



























