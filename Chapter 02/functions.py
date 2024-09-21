"""
The file will showcase functions in python
"""


# Define & Call Function
# ------------------------------------------------------------------------------
def greet(name):
    print(f"Hello, {name}!")


greet("Nayan")  # Calls the greet function with the argument "Nayan"


# Functions with paramaters
# ------------------------------------------------------------------------------
def add(x, y):
    return x + y


# x is 3, y is 5; result is 8
result = add(3, 5)

# OR We can call method by keyword arguments like as below
result = add(x=3, y=5)


# Functions having paramaters with default values
# ------------------------------------------------------------------------------
def power(x, y=2):
    return x**y


# y defaults to 2; result1 is 9
result1 = power(3)

# y is 4; result2 is 81
result2 = power(3, 4)

# OR other way to call any method is
# y is 3, x is 4; result2 is 64
result2 = power(y=3, x=4)


# Variable scope in function
# ------------------------------------------------------------------------------
x = 10


def my_function():
    x = 5  # This is a local variable
    print(x)  # Prints 5


my_function()
print(x)  # Prints 10
