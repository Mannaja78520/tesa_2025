#integer
a = 1
b = 2

#Floating Point
c = 0.5
d = 7.0

#String
text1 = "Hello"
text2 = 'Goodbye'

#Array
array1 = [1, 2, 4, 8, "text"]

#Array Indexing
#print(array1[0])

#Array Size
#print(len(array1))

#Matrix
matrix1 = [ [1, 0],
	[4, 5],
	[7, 8] ]

#Matrix Indexing
#print(matrix1[2][0])

#Matrix Size
#print("Row Size: ", len(matrix1))
#print("Column Size: ", len(matrix1[0]))


#Tuple (immutable)
tuple1 = ("text1", 2, 0.5, "text2")

#Tuple Indexing
#print(tuple1[0])

#Tuple Size
#print(len(tuple1))

#Dictionary (Key-Value Pairs)
dict1 = {"Brand": "iPhone", "Model": "17 Pro Max", "Year": 2025, "Price": "????"}

#Dictionary Indexing
#print(dict1["Brand"])

#Conditional Statements
"""
m = 1
if m == 1:
	print(1)
elif m == 2:
	print(100)
else:
	print(1000)
"""
#Beware of Floating Point


#Loops
#for
"""
for i in range(10):
	print(i)
"""
list = ["apple", "orange", "melon", "banana"]
"""
for item in list:
	print(item)
"""

#while
count = 10
"""
while count > 0:
	print(count)
	count -= 1
"""

#function
def function1():
	print("Function called")

#Call function
#function1()

#Parameters and Arguments
def function2(parameter1, parameter2):
	toPrint = parameter1 + parameter2
	print(toPrint)

argument1 = 2
argument2 = 10
#function2(argument1, argument2)

#Return Value
def function3(input1, input2):
	result = input1 * input2
	return result

returnValue = function3(2, 8)
#print(returnValue)


#Object-Oriented Programming

class Car:
	def __init__(self, model, color):
		self.model = model
		self.color = color
	def drive(self, speed):
		print(f"This car is moving at {speed} kmph")

car1 = Car("Ford", "Black")
#print(car1.model)
#car1.drive(120)

#Library Import
import time
import random

current_sec = time.time()
randomNumber = random.random()

#print(current_sec)
#print(randomNumber)


#File Handling
#Syntax ---> file = open("oldFile.txt", "mode")

with open("oldFile.txt", "r") as file:
	content = file.read()  # Reads the entire file content
	print(content)
"""
with open("newFile.txt", "w") as file:
	file.write("This is a new line.\n")
	file.write("Another line of text.")
"""