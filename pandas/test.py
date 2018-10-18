#Starttime 9PM
import csv

import json

testlst = ["William IV", "John III", "Jane IV", "John IV", "George VI", "Elizabeth III", "William II", "George XLVII", "George XXXIV", "Elizabeth I"]
lst = ["George VI", "William II", "Elizabeth I", "John XXXIX"]
lsttest = []
Romanlst = ["John XXI", "John XIX", "John XV", "John XII", "John IX", "John V", "John I", "John L", "John XLIX", "John XLV", "John XL", "John XXXIX", "John XXXVIII", "John XXVIII", "John XXV"]
romantuple = (("M",  1000),("CM", 900),("D",  500),("CD", 400),("CD", 400),("XC", 90),("C",  100),("L",  50),("XL", 40),("X",  10),("IX", 9),("V", 5),("IV", 4),("I",  1))
romandict = dict((x,y) for x,y in romantuple)

def roman_to_int(lst_of_numbers,troubleshoot=True, skip=False):
    index_numbers = range(len(lst_of_numbers))  #to keep track of the position of roman numeral
    lst_size = len(lst_of_numbers)
    tracker = 0
    for x in index_numbers:
        print(lst_size, "ls")
        if troubleshoot:
            print(x, "x")
        if skip:
            print(x,"skipped")
            skip = False
            continue
        if lst_size == 1:
            calculated_from_roman_to_numbers = lst_of_numbers[x] + tracker
            return calculated_from_roman_to_numbers

        if lst_size == 2:  # For 2 roman numerals only need to subtract
            if lst_of_numbers[x] < lst_of_numbers[x + 1]:
                calculated_from_roman_to_numbers = lst_of_numbers[x + 1] - lst_of_numbers[x] + tracker
                return calculated_from_roman_to_numbers
            else:
                calculated_from_roman_to_numbers = tracker + lst_of_numbers[x + 1] + lst_of_numbers[x]
                return calculated_from_roman_to_numbers
        if lst_size > 2:   #For 3 or more need to add and then subtract
            if lst_of_numbers[x] < lst_of_numbers[x+1]:
                    addition = lst_of_numbers[x+1] - lst_of_numbers[x]
                    print(addition, "add")
                    print(tracker, 'before1')
                    tracker = tracker + addition
                    print(tracker, 'after1')
                    skip = True
                    lst_size = lst_size - 2

            if lst_of_numbers[x] == lst_of_numbers[x+1]:
                print(lst_of_numbers[x], "add2")
                tracker = lst_of_numbers[x] + tracker
                print(tracker, 'after2')
                lst_size = lst_size - 1

            if lst_of_numbers[x] > lst_of_numbers[x+1]:
                print(lst_of_numbers[x+1], "add3")
                tracker = lst_of_numbers[x] + tracker
                print(tracker, 'after3')
                lst_size = lst_size - 1

def int_to_roman(num):
    roman_numerals = []
    for key, value in romandict.items():
        count = num // value
        num -= count * value
        roman_numerals.append(key * count)
    return ''.join(roman_numerals)




def getSortedList(lst):
    if all(isinstance(x, str) for x in lst):
        roman_to_int_convertedlst = []
        int_to_roman_convertedlst = []
        for name,ordinal in [royalname.split() for royalname in lst]: #split the name and roman number
            temp = []
            print(name,ordinal , "- orignal name and numeral")
            print(ordinal, "- orignal roman numeral")
            print(len(ordinal),"- number of roman numerals")
            if len(ordinal) > 1:
                for single_ordinal in list(ordinal):   #spilts ordinal into individual numerical pieces for conversion
                    print(single_ordinal)
                    single_ordinal = romandict[single_ordinal] #Converts Roman values to numbers
                    temp.append(single_ordinal)
                print(temp, "converted values")
                newnumber = roman_to_int(temp)  #Calculates Roman values
                print(newnumber, "number")
            if len(ordinal) == 1:                       #single ordinals dont need to be checked for subtraction
                newnumber = romandict[ordinal]
            royalname = name, newnumber
            roman_to_int_convertedlst.append(royalname)
        print(roman_to_int_convertedlst, "convertedlst")
        sortedlst = sorted(roman_to_int_convertedlst)
        print(sortedlst, "sorted")
        for name,integer in sortedlst:
            roman_numeral = int_to_roman(integer)
            int_to_roman_convertedlst.append(name+" "+str(roman_numeral))
    return int_to_roman_convertedlst

a = ['viatsko / awesome-vscode', 'jlaine / aiortc', 'Syllo / nvtop', 'imhuay / Algorithm_Interview_Notes-Chinese', '30-seconds / 30-seconds-of-code', 'Microsoft / MS-DOS', 'Mojang / brigadier', 'iovisor / bpftrace', 'Snailclimb / JavaGuide', 'kdn251 / interviews', 'heyscrumpy / tiptap', 'kamranahmedse / developer-roadmap', 'hasura / graphql-engine', 'TheAlgorithms / Python', 'MontFerret / ferret', 'CyC2018 / CS-Notes', 'Hacktoberfest-2018 / Hello-world', 'mit-pdos / biscuit', 'b3log / symphony', 'enochtangg / quick-SQL-cheatsheet', 'sourcegraph / sourcegraph', 'Mojang / DataFixerUpper', 'firstcontributions / first-contributions', 'lacuna / bifurcan', 'vuejs / vue']

b =['9,383', '630', '808', '6,139', '27,330', '10,829', '1,357', '292', '8,039', '27,527', '971', '58,274', '4,279', '13,120', '2,328', '37,831', '300', '175', '5,448', '1,830', '3,097', '526', '3,588', '341', '115,909']

c = ['JavaScript', 'Python', 'C', 'Python', 'JavaScript', 'Assembly', 'Java', 'C++', 'Java', 'Java', 'JavaScript', 'JavaScript', 'Python', 'Go', 'Java', 'Go', 'Java', 'Go', 'Java', 'Java', 'JavaScript']

import itertools

tuple = list(itertools.zip_longest(a, b, c))
#print(tuple)
#print(len(tuple))

#github_data = [('awesome-vscode', '9510', 'JavaScript'), ('Algorithm_Interview_Notes-Chinese', '6185', 'Python'), ('30-seconds-of-code', '27382', 'JavaScript'), ('nvtop', '869', 'C'), ('aiortc', '656', 'Python'), ('MS-DOS', '10870', 'Assembly'), ('bpftrace', '325', 'C++'), ('brigadier', '1395', 'Java'), ('JavaGuide', '8069', 'Java'), ('interviews', '27575', 'Java'), ('tiptap', '1010', 'JavaScript'), ('developer-roadmap', '58329', 'Null'), ('Python', '13149', 'Python'), ('biscuit', '204', 'Go'), ('CS-Notes', '37843', 'Null'), ('Hello-world', '318', 'Java'), ('ferret', '2343', 'Go'), ('graphql-engine', '4302', 'JavaScript'), ('quick-SQL-cheatsheet', '1848', 'Null'), ('symphony', '5458', 'Java'), ('bifurcan', '361', 'Java'), ('sourcegraph', '3114', 'Go'), ('first-contributions', '3598', 'Null'), ('DataFixerUpper', '544', 'Java'), ('vue', '115929', 'JavaScript')]

import tensorflow as tf

hello = tf.constant('Hello there!')

print(sess.run(hello))