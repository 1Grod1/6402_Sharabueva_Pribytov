import numpy as np

def RREF(A):
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
        #for j in range(r,0, -1):
         #   if A[j-1,lead]!=0:
          #      A[j-1]=A[j-1]^A[r]
    #Избавляемся от нулевых строк
    A = A[~np.all(A == 0, axis = 1)]
    return A

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

# Пример использования
A = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int)

rref_matrix,leadCol = Gresult(A)
rows, cols = rref_matrix.shape
Xresult = DelLead(rref_matrix,leadCol)
Matr = Hmatr(Xresult,leadCol,rows, cols)
print("\nФункция Gresult задание 1.3.1\n",rref_matrix)
print("\nзадание 1.3.2\n n = ",cols,"\nk = ",rows,"\nLead Col = ",leadCol)
print("\nФункция DelLead  задание 1.3.3\n",Xresult)
print("\nФункция Hmatr  задание 1.3.4\n",Matr)

