"""
The file will showcase loops in python
"""

# For loop
# ------------------------------------------------------------------------------

# Example - 1
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(f"I like {fruit}")

# Exaple - 2
for i in range(5):
    print(f"Count: {i}")

# Example - 3 - Nested Loop
for i in range(3):
    for j in range(2):
        print(f"({i}, {j})")

# Example - 4 - Loop with break and continue statements
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

print("Using 'break':")
for fruit in fruits:
    if fruit == "date":
        break  # Exit the loop when "date" is found
    print(f"I like {fruit}")

print("\nUsing 'continue':")
for fruit in fruits:
    if fruit == "date":
        continue  # Skip the iteration when "date" is found
    print(f"I like {fruit}")


# While loop
# ------------------------------------------------------------------------------

# Example - 1
count = 0

while count < 5:
    print(f"Count: {count}")
    count += 1  # Increment the count


# Example - 2 - Loop with break and continue statements
count = 0

while count < 5:
    if count == 2:
        break  # Exit the loop when count is 2
    elif count == 1:
        count += 1
        continue  # Skip the iteration when count is 1
    print(f"Count: {count}")
    count += 1  # Increment the count


# If Else statements
# ------------------------------------------------------------------------------
grade = 85

if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
elif grade >= 70:
    print("C")
else:
    print("D")


# One Line If Else Statment ................................
age = 20

status = "adult" if age >= 18 else "minor"
print(f"You are a {status}.")

# One Line Nested If Else Statment ................................
grade = 85

result = "A" if grade >= 90 else ("B" if grade >= 80 else ("C" if grade >= 70 else "D"))
print(f"Your grade is {result}.")
