import roman


testlst = ["William IV", "John III", "Jane IV", "John IV", "George VI", "Elizabeth III", "John II", "John III", "John VIII", "John VI", "John VII", "John XXI", "John XIX", "John XV", "John XII", "John IX", "John V", "John I", "John L", "John XLIX", "John XLV", "John XL", "John XXXIX", "John XXXVIII", "John XXVIII", "John XXV", "William II", "George XLVII", "George XXXIV", "Elizabeth I"]
lst = ["George VI", "William II", "Elizabeth I", "John XXXIX"]
Romanlst = ["John XXI", "John XIX", "John XV", "John XII", "John IX", "John V", "John I", "John II", "John III", "John XIII", "John L", "John XLIX", "John XLV", "John XL", "John XXXIX", "John XXXVIII", "John XXVIII", "John XXV"]


romantuple = (("M", 1000),("CM", 900),("D", 500),("CD", 400),("C", 100),("XC", 90), ("L", 50),("XL", 40),("X", 10),("IX", 9),("V", 5),("IV", 4),("I", 1))
romandict = dict((x,y) for x,y in romantuple)

def roman_to_int(lst_of_numbers, skip=False):

    """

    :param lst_of_numbers: input: List of numbers (values of each roman numeral)
    :param skip: Always False
    :return: Integer
    """


    index_numbers = range(len(lst_of_numbers))  #to keep track of the position of roman numeral
    lst_size = len(lst_of_numbers)              #reduces the complexity of handling multiple sizes
    tracker = 0                                 #Updates the current total
    for x in index_numbers:
        if skip:                                #allows the handling of pairwise roman numerals
            skip = False
            continue
        if lst_size == 1:
            calculated_from_roman_to_numbers = lst_of_numbers[x] + tracker
            return calculated_from_roman_to_numbers

        if lst_size == 2:
            if lst_of_numbers[x] < lst_of_numbers[x + 1]:
                calculated_from_roman_to_numbers = lst_of_numbers[x + 1] - lst_of_numbers[x] + tracker
                return calculated_from_roman_to_numbers
            else:
                calculated_from_roman_to_numbers = tracker + lst_of_numbers[x + 1] + lst_of_numbers[x]
                return calculated_from_roman_to_numbers

        if lst_size > 2:
            if lst_of_numbers[x] < lst_of_numbers[x+1]:
                    addition = lst_of_numbers[x+1] - lst_of_numbers[x]
                    tracker = tracker + addition
                    skip = True
                    lst_size = lst_size - 2

            if lst_of_numbers[x] >= lst_of_numbers[x+1]:
                tracker = lst_of_numbers[x] + tracker
                lst_size = lst_size - 1


def int_to_roman(num):

    """
    :param num: input: Integer
    :return: Roman numeral (character)
    """

    roman_numerals = []

    for key, value in romandict.items():
        count = num // value
        num -= count * value
        roman_numerals.append(key * count)

    return ''.join(roman_numerals)


def getSortedList(lst):

    """

    :param lst: List of Stringed RoyalNames
    :return: List of Sorted RoyalNames
    """



    if all(isinstance(x, str) for x in lst):
        roman_to_int_convertedlst = []
        int_to_roman_convertedlst = []

        for name,ordinal in [royalname.split() for royalname in lst]: #split the name and roman number
            temp = []

            if len(ordinal) > 1:
                for single_ordinal in list(ordinal):   #spilts ordinal into individual numerical pieces for conversion
                    single_ordinal = romandict[single_ordinal] #Converts Roman values to numbers
                    temp.append(single_ordinal)
                newnumber = roman_to_int(temp)  #Calculates Roman values
            if len(ordinal) == 1:                       #single ordinals dont need to be checked for subtraction
                newnumber = romandict[ordinal]
            royalname = name, newnumber                 #Places integers with original names
            roman_to_int_convertedlst.append(royalname)
        print("Converted from", lst, "\n"
              "to", roman_to_int_convertedlst)
        sortedlst = sorted(roman_to_int_convertedlst)   #sorts list (n log n)
        print("Sorted", sortedlst)

        for name,integer in sortedlst:                  #returns integers back into roman numerals
            roman_numeral = int_to_roman(integer)
            int_to_roman_convertedlst.append(name+" "+str(roman_numeral))

    return int_to_roman_convertedlst

def getSortedList_easy(lst):

    """
    Uses pre-made module Roman for speed comparision/ to check if conversion is correct
   :param lst: List of Stringed RoyalNames
    :return: List of Sorted RoyalNames
    """

    if all(isinstance(x, str) for x in lst):
        roman_to_int_convertedlst = []
        int_to_roman_convertedlst = []

        for name,ordinal in [royalname.split() for royalname in lst]: #split the name and roman number
            newnumber = roman.fromRoman(ordinal)
            royalname = name, newnumber
            roman_to_int_convertedlst.append(royalname)
        print("Converted from", lst, "\n"
              "to", roman_to_int_convertedlst)
        sortedlst = sorted(roman_to_int_convertedlst)
        print("Sorted", sortedlst)

        for name,integer in sortedlst:
            roman_numeral = roman.toRoman(integer)
            int_to_roman_convertedlst.append(name+" "+str(roman_numeral))

    return int_to_roman_convertedlst

print(getSortedList(testlst))