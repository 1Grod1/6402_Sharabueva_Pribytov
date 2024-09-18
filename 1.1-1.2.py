import numpy as np

def REF(A):
    """Реализовать функцию REF(), приводящую матрицу к
ступенчатому виду"""
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
                
        #if (max_row!=0 or A[k,r]!=0):
            #print(r, max_row, A[r,r])
            #k=k+1

    return A

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
        for j in range(r,0, -1):
            if A[j-1,lead]!=0:
                A[j-1]=A[j-1]^A[r]

    #Избавляемся от нулевых строк
    A = A[~np.all(A == 0, axis = 1)]
    
    return A


# Пример использования
A = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int)

ref_matrix = REF(A)
print("Функция REF\n",ref_matrix)
rref_matrix = RREF(A)
print("\nФункция RREF\n",rref_matrix)
