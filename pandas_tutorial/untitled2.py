#empty class
class my_fun:
    #pass
        print("Inside class")

print("Hello world")
my_fun


class employee:
    pass

emp_1=employee()
emp_2=employee()

print(emp_1)
print(emp_2)

emp_1.first_name="abc"
emp_1.last_name="xyz"
emp_1.email="abc.xyz@gmail.com"
emp_1.pay=5000


emp_2.first_name="def"
emp_2.last_name="wxy"
emp_2.email="def.wxy@gmail.com"
emp_2.pay=9000

print(emp_1.email)
print(emp_2.email)

#####

'''class employee:
    
    def __init__(self,first,last,pay):
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'   
    def fullname (self):
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):
        self.pay=int(self.pay*1.10)
        
emp_1=employee("abc","xyz",5000)
emp_2=employee("def","wxy",9000)

print(emp_1.fullname())
print(emp_2.fullname())

print(emp_1.first)
print(emp_2.last)

print(emp_1.email)
print(emp_2.email)

#function outside class
print('{} {}'.format(emp_1.first,emp_1.last))'''


class employee:
    raise_amount=1.10
    
    def __init__(self,first,last,pay):
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'   
    def fullname (self):
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):
        self.pay=int(self.pay*employee.raise_amount)
        

'''emp_1=employee("abc","xyz",5000)
emp_2=employee("def","wxy",9000)
        
print(emp_1.pay)
emp_1.apply_raise()
print(emp_1.pay)'''

print(employee.__dict__)


print(employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)



#####inheritence


class employee:
    raise_amount=1.10
    
    def __init__(self,first,last,pay):
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'   
    def fullname (self):
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):
        self.pay=int(self.pay*employee.raise_amount)
        
class Developer(employee):
    raise_amount=1.05
    
    def __init__(self,first,last,pay,prog_language):
        super().__init__(first,last,pay)
        #employee.__init__(self,first,last,pay)
        self.prog_language=prog_language
 
    
#superclass
'''dev_1=employee('abc','xyz',5000)
dev_2=employee('def','ghi',10000)

print(dev_1.email)
print(dev_2.email)'''

#derived class
dev_1=Developer('abc','xyz',4000,'Python' )
dev_2=Developer('def','ghi',10000,'Java')

print(dev_1.email)
print(dev_1.prog_language)


#Multiple Inheritance
class A:
    # variable of class A
    # functions of class A

class B:
    # variable of class A
    # functions of class A

class C(A, B):
    # class C inheriting property of both class A and B
    # add more properties to class C

#Multilevel Inheritance
class A:
    # properties of class A

class B(A):
    # class B inheriting property of class A
    # more properties of class B

class C(B):
    # class C inheriting property of class B
    # thus, class C also inherits properties of class A
    # more properties of class C









    