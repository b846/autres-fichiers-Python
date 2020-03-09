# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:04:46 2019

@author: Briac
"""


##### Utilisation d'un module
import turtle
turtle.forward(90)
turtle.done()

############################################################################
# Les séquences possibles sont
# Les listes : liste modifiable
address = ["Rue de la Loi", 16, 1000, "Bruxelles", "Belgique"]
numbers = [1, 2, 3, 4, 5]
print(numbers[0])
print(numbers[:3])
print(numbers[3:])
del(numbers[1:4])  # Suppression d'élements
numbers[0:0] = [0]  # Insertion d'éléments
essai = numbers + numbers # concaténation
print(essai)
a = [1, 2] * 4 # répétition
print(a)

i = -1
while i >= -len(numbers): 
    print(numbers[i])
    i -= 1

# Appartenance dans une liste
def contains(data, element):
    i = 0
    while i < len(data):
        if data[i] == element:
            return True
        i += 1
    return False

print(contains(a, 4))
print(4 in a)

print(not contains(a, 2))
print(2 not in a)


# Les tuples : liste non modifiable d'éléments
a = ()  # Tuple vide
# Une fonction peut renvoyer 2 éléments
t = 1, 2, 3       # emballage
a, b, c = t       # déballage


# Les autres types de séquences
# - Chaine de caractère : séquence non modifiable de caractères
print("pa" in "papa")
s = "pa" * 2
p = s + " est là."


# - Intervalle
i = range(1, 5)
# Les piles



"""
DICTIONARY
Why we need dictionary?

    It has 'key' and 'value'
    Faster than lists
    What is key and value. Example:
    dictionary = {'spain' : 'madrid'}
    Key is spain.
    Values is madrid.

    Lets practice some other properties like keys(), values(), update, add, check, remove key, remove all entries and remove dicrionary.
"""

#create dictionary and look its keys and values
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)



# Structure de données qui forment des séquences: liste, tuple et string
# Les dictionnaires sont des séquences non ordonnées
# Dans un dictionnaire, chaque clé est unique

# Liste comprehension
liste_2 = [i**2 for i in range(10)]

# Liste de liste
liste_3 = [[i for i in range(3)] for j in range(4)]

# Dict Comprehension
prenoms = ['Briac', 'Suliac', 'Gwenaelle', 'Nolwenn']
dico = {k:v for k, v in enumerate(prenoms)}

# zip
age = [24, 25, 12, 20]
dico_2 = {prenom:age for prenom, age in zip(prenoms, age)}
dico_3 = {prenom:age for prenom, age in zip(prenoms, age) if age >20}

# Tuple
tuple_1 = tuple((1, 2, 3))

# Format
print("il fait {} degree".format(25))

# Fonction open
#ecriture
f = open('fichier.txt', 'w')
f.write('bonjour') #écriture du fichier
f.close()

import os
os.getcwd()

# lecture
f = open('fichier.txt', 'r')
print(f.read())
f.close()

with open('fichier.txt', 'r') as f:
    f.read()