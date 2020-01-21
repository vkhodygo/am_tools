import json
import os
import sys
from itertools import product
from time import sleep
from zipfile import ZipFile, BadZipFile

import matplotlib.pyplot as plt
from numpy import multiply, concatenate, array, zeros, savetxt, mean, std, sqrt, linspace, float16
from fast_histogram import histogram2d
from pandas import read_csv, DataFrame, concat, io
from tqdm import tqdm
import multiprocessing

"""
Local package installation: python -m pip install --user -e am_tools/
"""


class GiantDensityFluctuations:
	"""
	This class provides functions for data loading and for actual calculations of giant density fluctuations.
	"""
	@staticmethod
	def load_raw_file(fp_, file_, chs=100000, ci=(2, 3), dt=float16, file_known=False, file_l=None, verbose=False):
		"""
		This function reads required data from the file_ or, when all data is zipped, from file in data.zip.
		:param fp_: file path template
		:param file_: file name
		:param chs: the size of a single data chunk
		:param ci: column indices
		:param dt: data type
		:param file_known: a parameter showing if there is any knowledge about the data file size
		:param file_l: if file_known is true, this is the file length
		:param verbose: if True the function describes the process of file loading, including possible errors
		:return: a pandas dataframe or None if there is no data to read/the data is corrupted
		"""
		file_path = fp_ % file_
		col_names = ('x', 'y')
		if not os.path.isfile(file_path):
			if not os.path.isfile(fp_ % 'data.zip'):
				verbose and print("\nNo (un)zipped data, skipping.\n")
				return None
			else:
				try:
					with ZipFile(fp_ % 'data.zip'):
						pass
				except BadZipFile:
					verbose and print("Corrupted zip, skipping. ")
					return None
				resulting_file = ZipFile(fp_ % 'data.zip').open(file_)
		else:
			try:
				read_csv(file_path, sep='\s+', engine='c', nrows=1)
			except io.common.EmptyDataError:
				verbose and print("\n%s is empty, skipping.\n" % file_)
				return None
			resulting_file = file_path

		verbose and print("\nLoading the file, this might take some time\n")
		df_, dt_ = DataFrame(), dict((x, dt) for x in col_names)
		if file_known:
			index = 0
			for x in read_csv(resulting_file, sep='\s+', engine='c', usecols=ci, names=col_names, dtype=dt_, chunksize=chs):
				df_ = concat([df_, x], ignore_index=True)
				index += 1
				sys.stdout.write("\rChunk %d/~%d" % (index, file_l))
		else:
			for x in read_csv(resulting_file, sep='\s+', engine='c', usecols=ci, names=col_names, dtype=dt_, chunksize=chs):
				df_ = concat([df_, x], ignore_index=True)
		verbose and print("\nDone.\n")
		return df_

	@staticmethod
	def density(population, domain_size, bin_size, verbose=False):
		"""
		This function calculates the average number density per bin
		:param population: number of particles
		:param domain_size: a tuple that contains the sizes of a rectangular domain
		:param bin_size: a tuple that contains the sizes of a single bin
		:param verbose: if True, the function prints an error message in the case of incorrect parameters
		:return: bins, an array containing the number of bins in x and y directions, average bin density,
		positions of bin edges for further 2d histogram construction
		"""
		if len(domain_size) != len(bin_size):
			verbose and print("\nIncorrect domain or binning parameters, exiting.\n")
			exit(1)
		else:
			bins = []
			factor = 1
			for i in range(len(domain_size)):
				f = int(domain_size[i] / bin_size[i])
				bins.append(f)
				factor /= f

			return factor * population, bins

	@staticmethod
	def density_fluctuations(x, y, av_density, range_, edges):
		"""
		This function calculates giant density fluctuations using provided particle positions and bin edges
		:param x: x coordinate of particles
		:param y: y coordinate of particles
		:param av_density: average bin density
		:param range_: the range of x, y coordinates. By default the lower bound is 0, thus only the upper bound is provided
		:param edges: bin edges
		:return: the normalized value of density fluctuations
		"""
		h = histogram2d(x, y, range=[[0, range_[0]], [0, range_[1]]], bins=[edges[0], edges[1]])
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
			"max_range": 100,
			"samples": 10000}
		self.verbose = False
		self.cluster = False
		self.samples = None

	def get_parameters(self):
		"""
		All parameters for the simulations are provided as an external JSON string,
		actual usage looks like this:
		echo  '{"path": "/path/%s"}' | python script.py
		:return: None
		"""
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
		self.samples = sys_par["samples"] if "samples" in sys_par else self.default_values["samples"]
		
		if self.cluster:
			self.min_range = sys_par["min_range"] if "min_range" in sys_par else self.default_values["min_range"]
			self.max_range = sys_par["max_range"] if "max_range" in sys_par else self.default_values["max_range"]
		return None

	@staticmethod
	def load_additional_parameters():
		"""
		Basic function that retuns bin sizes for the default confinement geometry
		:return: two arrays, each of them contains bin sizes in the corresponding direction
		"""
		# TODO replace this function
		f1 = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.5]
		f2 = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
		xbinsizes = multiply(f1, 120)
		ybinsizes = multiply(f2, 12)
		xbinsizes = concatenate((xbinsizes, array([0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30])))
		ybinsizes = concatenate((ybinsizes, array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6])))
		return xbinsizes, ybinsizes

	def general_sub_pipeline(self, df_, xb, yb):
		"""
		This function goes through the provided number of samples and calculates density fluctuations for
		all provided bin sizes
		:param df_: a pandas dataframe
		:param xb: bin sizes in x direction
		:param yb: bin sizes in y direction
		:return: an array containing the resulting data
		"""
		data = zeros((self.samples, len(xb) * len(yb)))
		x_v, y_v = df_['x'].to_numpy(), df_['y'].to_numpy()  # turn to numpy first, it's ~2 times faster
		iterator = tqdm(product(xb, yb), total=len(xb) * len(yb)) if self.verbose else product(xb, yb)

		count, domain_size_t = 0, (self.size_x, self.size_y)
		for x_, y_ in iterator:
			av_d, bins_t = self.density(population=self.pop, domain_size=domain_size_t, bin_size=(x_, y_))
			for i in range(self.samples):
				i_min, i_max = i * self.pop, (i + 1) * self.pop
				data[i, count] = self.density_fluctuations(x_v[i_min:i_max], y_v[i_min:i_max], av_d, domain_size_t, bins_t)
			count += 1

		self.verbose and sleep(1)

		proc_data = zeros((5, len(xb) * len(yb)))
		c_ = 0
		self.verbose and print("%-8.s %-8.s %-8.s %-8.s %-8.s" % ("x_bin", "y_bin", "density", "mean", "std"))
		for x_, y_ in product(xb, yb):
			d, _ = self.density(population=self.pop, domain_size=domain_size_t, bin_size=(x_, y_))
			m, s = mean(data[:, c_]), std(data[:, c_])
			self.verbose and print("%-8.2f %-8.2f %-8.2f %-8.2f %-8.2e" % (x_, y_, d, m, s))
			proc_data[0, c_], proc_data[1, c_], proc_data[2, c_], proc_data[3, c_], proc_data[4, c_] = x_, y_, d, m, s
			c_ += 1
		return proc_data

	@staticmethod
	def data_reduce(data):
		pass

	@staticmethod
	def extract_exponent(data):
		pass

	@staticmethod
	def save_proc_data(data, save_path):
		"""
		Saves the resulting array.
		:param data: array name
		:param save_path: path to the file
		:return: None
		"""
		savetxt(save_path % "gdf_processed_data.txt", data.T, fmt='%.4e')
		return None

	@staticmethod
	def plot_figure(data, save_path):
		"""
		Plots a gdf figure and saves it.
		:param data: data to plot
		:param save_path: path to the figure
		:return: None
		"""
		plt.figure()
		plt.yscale('log')
		plt.xscale('log')
		plt.errorbar(data[2, :], data[3, :], yerr=data[4, :], fmt='o')
		plt.savefig(save_path % "plot.png")
		plt.close()
		return None

	def serial_data_pipeline(self):
		"""
		This function is a pipeline for serial data processing.
		It gets parameters, collects and processes the corresponding data.
		:return: None
		"""
		self.get_parameters()
		xb, yb = self.load_additional_parameters()
		df = self.load_raw_file(self.data_path, self.fn, verbose=self.verbose)
		data = self.general_sub_pipeline(df, xb, yb)
		self.save_proc_data(data, self.data_path)
		self.plot_figure(data, self.data_path)
		return None

	def parallel_data_pipeline_single_file(self):
		pass

	def parallel_data_pipeline_multiple_files(self):
		pass
