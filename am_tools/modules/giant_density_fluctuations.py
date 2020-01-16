import json
import os
import sys
from itertools import product
from time import sleep
from zipfile import ZipFile, BadZipFile

import matplotlib.pyplot as plt
from numpy import multiply, concatenate, array, zeros, savetxt, mean, std, sqrt, linspace, histogram2d, float16
from pandas import read_csv, DataFrame, concat, io
from tqdm import tqdm


class GiantDensityFluctuations:
	@staticmethod
	def load_file(fp_, file_, chs=100000, ci=(2, 3), cn=('x', 'y'), dt=float16, file_known=False, file_l=None, v=False):
		file_path = fp_ % file_

		if not os.path.isfile(file_path):
			if not os.path.isfile(fp_ % 'data.zip'):
				print("\nNo (un)zipped data, skipping.\n" % file_) if v else None
				return None
			else:
				try:
					with ZipFile(fp_ % 'data.zip') as zf:
						pass
				except BadZipFile:
					print("Corrupted zip, skipping. ") if v else None
					return None
				resulting_file = ZipFile(fp_ % 'data.zip').open(file_)
		else:
			try:
				read_csv(file_path, sep='\s+', engine='c', nrows=1)
			except io.common.EmptyDataError:
				print("\n%s is empty, skipping.\n" % file_) if v else None
				return None
			resulting_file = file_path

		print("\nLoading the file, this might take some time\n") if v else None
		df_, dt_ = DataFrame(), dict((x, dt) for x in cn)
		if file_known:
			index = 0
			for x in read_csv(resulting_file, sep='\s+', engine='c', usecols=ci, names=cn, dtype=dt_, chunksize=chs):
				df_ = concat([df_, x], ignore_index=True)
				index += 1
				sys.stdout.write("\rChunk %d/~%d" % (index, file_l))
		else:
			for x in read_csv(resulting_file, sep='\s+', engine='c', usecols=ci, names=cn, dtype=dt_, chunksize=chs):
				df_ = concat([df_, x], ignore_index=True)
		print("\nDone.\n") if v else None
		return df_

	@staticmethod
	def density(population, domain_size, bin_size, verbose=False):
		if len(domain_size) != len(bin_size):
			print("\nIncorrect domain or binning parameters, exiting.\n") if verbose else None
			exit(1)
		else:
			bins = []
			factor = 1
			for i in range(len(domain_size)):
				f = int(domain_size[i] / bin_size[i])
				bins.append(f)
				factor /= f
			xedges = linspace(0, domain_size[0], bins[0] + 1)
			yedges = linspace(0, domain_size[1], bins[1] + 1)
			return bins, factor * population, (xedges, yedges)

	@staticmethod
	def den_fl(x, y, av_density, edges):
		h, _, _ = histogram2d(x, y, bins=(edges[0], edges[1]))
		h -= av_density
		return sqrt(mean(multiply(h, h))) / sqrt(av_density)


class GDFanalysis(GiantDensityFluctuations):
	def __init__(self):
		self.pop = None
		self.size_x = None
		self.size_y = None
		self.data_path = None
		self.min_range = None
		self.max_range = None
		self.fn = None
		self.default_values = {
			"size_x": 120.0,
			"size_y": 12.0,
			"population": 961,
			"path": None,
			"filename": "simulation.main.data.bin",
			"verbose": False,
			"cluster": False,
			"min_range": 1,
			"max_range": 100}
		self.verbose = False
		self.cluster = False
		self.samples = 80000

	def get_parameters(self):
		try:
			sys_par = json.load(sys.stdin)
		except json.JSONDecodeError:
			print('Exiting on Invalid JSON format')
			sys.exit()

		if "path" in sys_par:
			self.data_path = sys_par["path"]
		else:
			print("\nNo data path provided, exiting.")
			sys.exit()

		self.pop = sys_par["population"] if "population" in sys_par else self.default_values["population"]
		self.size_x = sys_par["size_x"] if "size_x" in sys_par else self.default_values["size_x"]
		self.size_y = sys_par["size_y"] if "size_y" in sys_par else self.default_values["size_y"]
		self.fn = sys_par["filename"] if "filename" in sys_par else self.default_values["filename"]
		self.verbose = sys_par["verbose"] if "verbose" in sys_par else self.default_values["verbose"]
		self.cluster = sys_par["cluster"] if "cluster" in sys_par else self.default_values["cluster"]

		if self.cluster:
			self.min_range = sys_par["min_range"] if "min_range" in sys_par else self.default_values["min_range"]
			self.max_range = sys_par["max_range"] if "max_range" in sys_par else self.default_values["max_range"]
		return None

	@staticmethod
	def load_additional_parameters():
		# TODO replace this function
		f1 = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.5]
		f2 = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
		xbinsizes = multiply(f1, 120)
		ybinsizes = multiply(f2, 12)
		xbinsizes = concatenate((xbinsizes, array([0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30])))
		ybinsizes = concatenate((ybinsizes, array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6])))
		return xbinsizes, ybinsizes

	def general_sub_pipeline(self, df_, xb, yb):
		data = zeros((self.samples, len(xb) * len(yb)))
		x_v, y_v = df_['x'].to_numpy(), df_['y'].to_numpy()  # turn to numpy first, it's ~2 times faster
		iterator = tqdm(product(xb, yb), total=len(xb) * len(yb)) if self.verbose else product(xb, yb)

		count = 0
		for x_, y_ in iterator:
			bins, av_d, edges_t = self.density(population=self.pop, domain_size=(self.size_x, self.size_y), bin_size=(x_, y_))
			for i in range(self.samples):
				i_min, i_max = i * self.pop, (i + 1) * self.pop
				data[i, count] = self.den_fl(x_v[i_min:i_max], y_v[i_min:i_max], av_d, edges_t)
			count += 1

		sleep(1) if self.verbose else None

		proceeded_data = zeros((3, len(xb) * len(yb)))
		count = 0
		print("%-8.s %-8.s %-8.s %-8.s %-8.s" % ("x_bin", "y_bin", "density", "mean", "std")) if self.verbose else None
		for x_, y_ in product(xb, yb):
			_, d, _ = self.density(population=self.pop, domain_size=(self.size_x, self.size_y), bin_size=(x_, y_))
			m, s = mean(data[:, count]), std(data[:, count])
			print("%-8.2f %-8.2f %-8.2f %-8.2f %-8.2e" % (x_, y_, d, m, s)) if self.verbose else None
			proceeded_data[0, count], proceeded_data[1, count], proceeded_data[2, count] = d, m, s
			count += 1
		return proceeded_data

	@staticmethod
	def data_reduce(data):
		pass

	@staticmethod
	def extract_exponent(data):
		pass

	@staticmethod
	def save_proc_data(data, save_path):
		savetxt(save_path % "gdf_test.txt", data.T, fmt='%.2e')
		return None

	@staticmethod
	def plot_figure(data, save_path):
		plt.figure()
		plt.yscale('log')
		plt.xscale('log')
		plt.errorbar(data[0, :], data[1, :], yerr=data[2, :], fmt='o')
		plt.savefig(save_path % "plot.png")
		return None

	def serial_data_pipeline(self):
		self.get_parameters()
		xb, yb = self.load_additional_parameters()
		df = self.load_file(self.data_path, self.fn, v=self.verbose)
		data = self.general_sub_pipeline(df, xb, yb)
		self.save_proc_data(data, self.data_path)
		self.plot_figure(data, self.data_path)
		return None

	def parallel_data_pipeline(self):
		pass
