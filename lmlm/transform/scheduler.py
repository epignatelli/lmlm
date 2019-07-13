from multiprocessing import Pool


class Scheduler(object):
	def __init__(self,
				 n_workers=1):
		self.n_workers = n_workers

	def map(self, args):
		with Pool(n_workers) as p:
			p.map()
		pass

	def reduce(self, **kwargs):
		pass