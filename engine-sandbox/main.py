from ananas import *

E = Engine()
F = Engine(foo = 1000, bar = 1000)

Engine.foo = -1
Engine.bar = -1

G = Engine()

print('E', E.fun(1))
print('F', F.fun(1))
print('G', G.fun(1))
