#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:25:40 2020

@author: b
"""

print('Hello World!')
# \\Backslash
# \'Guillemet simple (apostrophe)
# \"Guillemet double
# \nSaut de ligne
# \rRetour chariot
# \tTabulation horizontale

print('Né le :', end=' ')
print(4, 8, 1961, sep='/')

firstname = input('Quel est ton prénom ? ')
print('Bonjour', firstname, 'et bienvenue !')


######### Type booléen
v = True
print(v)
print(type(v))
 

# Interruption de boucle
n = 1
while n <= 1000000:
    if n % 38 == 0 and n % 46 == 0:
        break
    n += 1
print(n, "est le plus petit nombre divisible par 38 et 46")
    
# Break permet de sorir de la boucle While
# Continue permet de revenir directement à l'exécution de la cdt while


def table(base, start=1, length=10):
    n = start
    while n < start + length:
        print(n, "x", base, "=", n * base)
        n += 1
table(4, length=2)


def pow(a, n):   #a puissance n
    if n == 1:
        return a
    if n % 2 == 0:
        return pow(a * a, n / 2)
    return a * pow(a * a, (n - 1) / 2)

############################################################
# ALGORITHME
# Compte le nombre de diviseurs stricts d'un nombre naturel non nul
# Pre  : n est un entier strictement positif
# Post : la valeur renvoyée contient le nombre de diviseurs stricts
#        de n
def nbdivisors(n):
    result = 0
    for i in range(1, n):
        if n % i == 0:
            result += 1
    return result

print('Le nombre 42 possède', nbdivisors(42), 'diviseurs stricts.')

def fibo(n):
    if n == 1 or n == 2:
        return 1
    return fibo(n-1) + fibo(n-2)

#Recherche d'une sous séquence
def issubsequence(subseq, seq):
    n = len(subseq)
    for i in range(0,len(seq)-n+1):
        if seq[i:i+n] == subseq:
            return True
    return False

#Nombre de voyelles
def nbvowels(s):
    result = 0
    for c in s:
        if c in 'aeiou':
            result += 1
    return result


# Logique
print(False & True) # AND
print(False | True) # OR
print(False ^ True) #XOR

f = lambda x: x**2
f(3)
f = lambda x, y: x**2 + y
f(3,2)
