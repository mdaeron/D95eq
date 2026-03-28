class Engine():

	foo = 2.
	bar = 0.1

	def __init__(self, foo = None, bar = None):
		self.foo = Engine.foo if foo is None else foo
		self.bar = Engine.bar if bar is None else bar

	def fun(self, x):
		return self.foo * x + self.bar
