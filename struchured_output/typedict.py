from typing import TypedDict

class Person(TypedDict):
    name = str
    age = int

new_person = Person = {'name':'kuldip','age':20}
print(new_person)