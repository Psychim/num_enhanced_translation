import time
class TicTocTimer:
	def __init__(self):
		self.sum_time=0
		self.last_time=0
	def tic(self):
		self.last_time=time.time()
	def toc(self):
		t=time.time()-self.last_time
		self.sum_time+=t
		self.last_time=0
		return t
	def get(self):
		return self.sum_time
	def reset(self):
		self.sum_time=0
		self.last_time=0