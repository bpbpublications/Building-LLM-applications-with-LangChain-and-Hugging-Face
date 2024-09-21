"""
The file will showcase different OOPs oncepts available in Python.
"""


# Define Class
# ------------------------------------------------------------------------------
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(f"{self.name} says Woof!")


# Define objects
my_dog = Dog("Buddy")
my_dog.bark()


# Class Inheritance
# ------------------------------------------------------------------------------
class Animal:
    def __init__(self, name):
        self.name = name


class Dog(Animal):
    def speak(self):
        print(f"{self.name} says Woof!")


# Here Buddy! is the name. As Dog inherits property of Animal class
# We are providing the name which will be utilized by Animal class
my_dog = Dog("Buddy!")
my_dog.speak()


# Class Encapsulation
# ------------------------------------------------------------------------------
class MyClass:
    def __init__(self):
        self._protected_var = 42


# Polymorphism
# ------------------------------------------------------------------------------
class Cat:
    def speak(self):
        print("Meow!")


def make_animal_speak(animal):
    animal.speak()


# Define objetcs
my_cat = Cat()
make_animal_speak(my_cat)


# Abstraction
# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.1415 * self.radius**2


# Method overriding
# ------------------------------------------------------------------------------
class Animal:
    def speak(self):
        print("Generic animal sound")


class Dog(Animal):
    def speak(self):
        print("Woof!")


# Define objects
my_dog = Dog()
my_dog.speak()
