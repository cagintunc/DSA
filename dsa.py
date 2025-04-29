

def heapify_up(arr, i): #recursive max heap
    parent = (i - 1) / 2
    if parent > 0 and arr[i] > arr[parent]:
        arr[parent], arr[i] = arr[i], arr[parent]
        heapify_up(arr, parent)
    else: return 

    
def heapify_down(arr, i): #recursive max heap
    largest = i
    left = 2*i+1
    right = 2*i+2
    if left < len(arr) and arr[left] > arr[largest]:
        largest = left
    if right < len(arr) and arr[right] > arr[largest]:
        largest = right
    if i != largest:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_down(arr, largest)
    else:
        return 


def heapify_up_iter(arr, i): # max heapify iterative 
    parent = (i - 1) / 2
    while(True):
        if parent > 0 and arr[parent] < arr[i]:
            arr[i], arr[parent] = arr[parent], arr[i]
            i = parent
            parent = (i - 1) / 2
        else:
            break

def heapify_down_iter(arr, i):
    while(True):
        largest = i
        left = 2*i + 1
        right = 2*i + 2
        if left < len(arr) and arr[largest] < arr[left]:
            largest = left
        if right < len(arr) and arr[largest] < arr[right]:
            largest = right
        if largest != i:
            arr[largest], arr[i] = arr[i], arr[largest]
            i = largest
        else:
            break


