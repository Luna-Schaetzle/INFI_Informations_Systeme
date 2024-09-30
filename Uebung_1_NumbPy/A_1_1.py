import array
import random

# Create an integer array named ranNum
ranNum = array.array('i', [random.randint(0, 100) for i in range(10)])

for i in range(10):
    print(ranNum[i])


