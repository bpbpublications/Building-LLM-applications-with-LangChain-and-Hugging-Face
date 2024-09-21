"""
The file will showcase class in python
"""


# Define Class
# ------------------------------------------------------------------------------
class MyClass:
    attribute1 = 0
    attribute2 = "Hello"

    def method1(self):
        pass

    def method2(self, parameter):
        pass


# Create instances of MyClass  i.e. objects of class
obj1 = MyClass()
obj2 = MyClass()

# Define attribues i.e. properties
obj1.attribute1 = 42
obj2.attribute2 = "World"


# Define and Call Class and Methods
# ------------------------------------------------------------------------------
class MyClass:
    def say_hello(self):
        print("Hello, world!")


obj = MyClass()
obj.say_hello()  # Calls the say_hello method


# Self Method
# ------------------------------------------------------------------------------
class MyClass:
    def set_attribute(self, value):
        self.attribute1 = value

    def get_attribute(self):
        return self.attribute1


obj = MyClass()
obj.set_attribute(42)
value = obj.get_attribute()  # Retrieves the value


# Constructor Method i.e. __init__
# ------------------------------------------------------------------------------
class MyClass:
    def __init__(self, initial_value):
        self.attribute1 = initial_value


obj = MyClass(42)  # Creates an object with an initial value of 42


# Class example - How it will look a like.
# ------------------------------------------------------------------------------
class Person:
    """
    A class to represent a person.
    This class provides a simple way to store and retrieve information about a person.

    Attributes:
    name (str): The name of the person.
    age (int): The age of the person.
    """

    def __init__(self, name, age):
        """Initializes a new Person object.

        Args:
            name (str): The name of the person.
            age (int): The age of the person.
        """
        self.name = name
        self.age = age

    def greet(self):
        """
        Prints a friendly greeting message.
        Returns:
            str: A greeting message.
        """
        return f"Hello, my name is {self.name}, and I am {self.age} years old."


# Creating an instance of the Person class
person1 = Person("Nayan", 35)

# Calling the greet method
greeting = person1.greet()
print(greeting)
