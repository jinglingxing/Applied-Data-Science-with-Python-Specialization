
# Slicing

x = '0123456789'

print(x[0])
print(x[-1]) #backward

print(x[2:5]) 
print(x[2:6])
print(x[:3]) #from begining
print(x[6:]) #till the end
print(x[-4:-2]) #backward


# string operators

# Concatenate
firstname = 'Jin Ling'
lastname = 'XING'

print('frenchway', firstname+' '+lastname, 'chineseway', lastname+' '+firstname)

# multiplication
message = 'love '
print(message*4,'youuuuu','infinity'+' much')

# dictionaries
# see https://www.tutorialspoint.com/python/python_hash_table


# unpack tupples

def sum_and_minus(a,b):
    c=a+b
    d=b-a
    return (c,d)

s,m=sum_and_minus(3,4)
print(s,m)



# use of format

x = 'item : {}, price : {} {}'.format('banana',3.25,'CAD$')
print(x)


# Class

class Point:
    
    def __init__(self, x, y, z=0):
        self.x=x
        self.y=y
        self.z=z
        
    def distance(self, Pt):
        return Pt.x-self.x+Pt.y-self.y+Pt.z-self.z
    
    def setztozero(self):
        self.z=0
    
    def __str__(self):
        return 'x : {}, y : {}, z : {}'.format(self.x, self.y, self.z)
    
a = Point(2,3,4)
b = Point(2,5,4)
print(a.distance(b))

a.setztozero()
print(a.distance(b))

# map usage

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
  s=person.split(' ')
  return s[0]+' '+s[-1]#Your answer here

list(map(split_title_and_name, people))#Your answer here)

# other solution using format

def split_title_and_name(person):
    title = person.split()[0]
    lastname = person.split()[-1]
    return '{} {}'.format(title, lastname)

list(map(split_title_and_name, people))

# Lambda

my_function = lambda a, b, c : a + b #no default values, no complex logic
print(my_function(1,2,3))

# Example

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

#option 1
#my_function = lambda person: person.split()[0] + ' ' + person.split()[-1]

#for person in people:
#    print(split_title_and_name(person) == (my_function(person) ))

#option 2
list(map(split_title_and_name, people)) == list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))

# list comprehension

my_list = [number for number in range(0,50) if number % 2 == 0]
print(my_list)

# example

def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

times_tables() == [i*j for j in range(10) for i in range(10) ]

# Here’s a harder question which brings a few things together.

# Many organizations have user ids which are constrained in some way. Imagine you work at an internet service provider and the user ids are all two letters followed by two numbers (e.g. aa49). Your task at such an organization might be to hold a record on the billing activity for each possible user.

# Write an initialization line as a single list comprehension which creates a list of all possible user ids. Assume the letters are all lower case.


lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'

answer = [i+k+j+l for i in lowercase for k in lowercase for j in digits for l in digits ]
correct_answer == answer


