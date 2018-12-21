"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import numpy
from sklearn import naive_bayes
from sklearn import tree
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase


class PatternGraphs:
	"""
	This template class is used to define a task for the gSpan implementation.
	You should not modify this class but extend it to define new tasks
	"""

	def __init__(self, database):
		# A list of subsets of graph identifiers.
		# Is used to specify different groups of graphs (classes and training/test sets).
		# The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
		# in which the examined pattern is present.
		self.gid_subsets = []

		self.database = database  # A graphdatabase instance: contains the data for the problem.

	def store(self, dfs_code, gid_subsets):
		"""
		Code to be executed to store the pattern, if desired.
		The function will only be called for patterns that have not been pruned.
		In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
		:param dfs_code: the dfs code of the pattern (as a string).
		:param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
		"""
		print("Please implement the store function in a subclass for a specific mining task!")

	def prune(self, gid_subsets):
		"""
		prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
		should be pruned.
		:param gid_subsets: A list of the cover of the pattern for each subset.
		:return: true if the pattern should be pruned, false otherwise.
		"""
		print("Please implement the prune function in a subclass for a specific mining task!")


class FrequentPositiveGraphs(PatternGraphs):
	"""
	Finds the frequent (support >= minsup) subgraphs among the positive graphs.
	This class provides a method to build a feature matrix for each subset.
	"""

	def __init__(self, minsup, database, subsets):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		"""
		super().__init__(database)
		self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
		self.minsup = minsup
		self.gid_subsets = subsets

	# Stores any pattern found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		self.patterns.append((dfs_code, gid_subsets))

	# Prunes any pattern that is not frequent in the positive class
	def prune(self, gid_subsets):
		# first subset is the set of positive ids
		return len(gid_subsets[0]) < self.minsup

	# creates a column for a feature matrix
	def create_fm_col(self, all_gids, subset_gids):
		subset_gids = set(subset_gids)
		bools = []
		for i, val in enumerate(all_gids):
			if val in subset_gids:
				bools.append(1)
			else:
				bools.append(0)
		return bools

	# return a feature matrix for each subset of examples, in which the columns correspond to patterns
	# and the rows to examples in the subset.
	def get_feature_matrices(self):
		matrices = [[] for _ in self.gid_subsets]
		for pattern, gid_subsets in self.patterns:
			for i, gid_subset in enumerate(gid_subsets):
				matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
		return [numpy.array(matrix).transpose() for matrix in matrices]


class ConfidencePositiveGraphs(PatternGraphs):

	def __init__(self, k, minsup, database, subsets):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		"""
		super().__init__(database)
		self.patterns = {}  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
		self.k = k
		self.minsup = minsup
		self.gid_subsets = subsets

	# Stores any pattern found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		print(gid_subsets)
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[1])
		sum_sup = pos_sup + neg_sup
		confidence = pos_sup / sum_sup
		keys = self.patterns.keys()
		if len(keys) < self.k:
			min_conf = -1
		else:
			conf_list = [key[0] for key in keys]
			min_conf = min(conf_list)
		if (confidence < min_conf) or (sum_sup < self.minsup):
			return 0
		try:
			self.patterns[(confidence, sum_sup)].append((dfs_code, gid_subsets))
		except:
			self.patterns[(confidence, sum_sup)] = [(dfs_code, gid_subsets)]
			if (len(self.patterns) > self.k):
				keys = self.patterns.keys()
				conf_list = [key[0] for key in keys]
				min_conf = min(conf_list)
				sup_list = [key[1] for key in keys if key[0] == min_conf]
				min_sup = min(sup_list)
				self.patterns.pop((min_conf, min_sup))
		# print(keys)

	# Prunes any pattern that is not frequent in the positive class
	def prune(self, gid_subsets):
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[1])
		sum_sup = pos_sup + neg_sup
		return sum_sup < self.minsup


class ConfidencePositiveGraphs2(PatternGraphs):

	def __init__(self, k, minsup, database, subsets):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		"""
		super().__init__(database)
		self.patterns = {}  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
		self.k = k
		self.minsup = minsup
		self.gid_subsets = subsets

	# Stores any pattern found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		# print(gid_subsets)
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[2])
		sum_sup = pos_sup + neg_sup
		confidence = pos_sup / sum_sup
		keys = self.patterns.keys()
		if len(keys) < self.k:
			min_conf = -1
		else:
			conf_list = [key[0] for key in keys]
			min_conf = min(conf_list)
		if (confidence < min_conf) or (sum_sup < self.minsup):
			return 0
		try:
			self.patterns[(confidence, sum_sup)].append((dfs_code, gid_subsets))
		except:
			self.patterns[(confidence, sum_sup)] = [(dfs_code, gid_subsets)]
			if (len(self.patterns) > self.k):
				keys = self.patterns.keys()
				conf_list = [key[0] for key in keys]
				min_conf = min(conf_list)
				sup_list = [key[1] for key in keys if key[0] == min_conf]
				min_sup = min(sup_list)
				self.patterns.pop((min_conf, min_sup))
		# print(keys)

	# Prunes any pattern that is not frequent in the positive class
	def prune(self, gid_subsets):
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[2])
		sum_sup = pos_sup + neg_sup
		return sum_sup < self.minsup

	def create_fm_col(self, all_gids, subset_gids):
		subset_gids = set(subset_gids)
		bools = []
		for i, val in enumerate(all_gids):
			if val in subset_gids:
				bools.append(1)
			else:
				bools.append(0)
		return bools

	def get_feature_matrices(self):
		matrices = [[] for _ in self.gid_subsets]
		keys = self.patterns.keys()
		for key in keys:
			for pattern, gid_subsets in self.patterns[key]:
				for i, gid_subset in enumerate(gid_subsets):
					matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
		return [numpy.array(matrix).transpose() for matrix in matrices]


class ConfidencePositiveGraphs3(PatternGraphs):

	def __init__(self, k, minsup, database, subsets):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		"""
		super().__init__(database)
		self.patterns = {}  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
		self.k = k
		self.minsup = minsup
		self.gid_subsets = subsets

	# Stores any pattern found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		# print(type(dfs_code))
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[2])
		sum_sup = pos_sup + neg_sup
		pos_conf = pos_sup / sum_sup
		neg_conf = neg_sup / sum_sup
		if pos_conf >= neg_conf:
			confidence = pos_conf
			flag = 'pos'
		else:
			confidence = neg_conf
			flag = 'neg'
		keys = self.patterns.keys()
		if len(keys) < self.k:
			min_conf = -1
		else:
			conf_list = [key[0] for key in keys]
			min_conf = min(conf_list)
		if (confidence < min_conf) or (sum_sup < self.minsup):
			return 0
		try:
			self.patterns[(confidence, sum_sup)].append((dfs_code, gid_subsets, flag))
		except:
			self.patterns[(confidence, sum_sup)] = [(dfs_code, gid_subsets, flag)]
			if (len(self.patterns) > self.k):
				keys = self.patterns.keys()
				conf_list = [key[0] for key in keys]
				min_conf = min(conf_list)
				sup_list = [key[1] for key in keys if key[0] == min_conf]
				min_sup = min(sup_list)
				self.patterns.pop((min_conf, min_sup))
		# print(keys)

	# Prunes any pattern that is not frequent in the positive class
	def prune(self, gid_subsets):
		pos_sup = len(gid_subsets[0])
		neg_sup = len(gid_subsets[2])
		sum_sup = pos_sup + neg_sup
		return sum_sup < self.minsup


def example1():
	"""
	Runs gSpan with the specified positive and negative graphs, finds all frequent subgraphs in the positive class
	with a minimum positive support of minsup and prints them.
	"""

	a = 11
	if a == 1:
		args = sys.argv
		database_file_name_pos = args[1]  # First parameter: path to positive class file
		database_file_name_neg = args[2]  # Second parameter: path to negative class file
		k = int(args[3])
		minsup = int(args[4])  # Third parameter: minimum support
	else:
		database_file_name_pos = 'data/molecules-small.pos'
		database_file_name_neg = 'data/molecules-small.neg'
		k = 5
		minsup = 5


	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids
	subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
	print(subsets)
	task = ConfidencePositiveGraphs(k, minsup, graph_database, subsets)  # Creating task

	gSpan(task).run()  # Running gSpan

	# Printing frequent patterns along with their positive support:
	keys = task.patterns.keys()
	for key in keys:
		for pattern, a in task.patterns[key]:
			confidence = key[0]
			support = key[1] # This will have to be replaced by the confidence and support on both classes
			print('{} {} {}'.format(pattern, confidence, support))
			print(a)


def example2():
	"""
	Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
	the positive class with a minimum support of minsup.
	Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
	the test set.
	Performs a k-fold cross-validation.
	"""
	a = 11

	if a == 1:
		args = sys.argv
		database_file_name_pos = args[1]  # First parameter: path to positive class file
		database_file_name_neg = args[2]  # Second parameter: path to negative class file
		k  = int(args[3])
		minsup = int(args[4])  # Third parameter: minimum support (note: this parameter will be k in case of top-k mining)
		nfolds = int(args[5])  # Fourth parameter: number of folds to use in the k-fold cross-validation.
	else:
		database_file_name_pos = 'data/molecules-small.pos'
		database_file_name_neg = 'data/molecules-small.neg'
		k = 5
		minsup = 5
		nfolds = 4

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	# If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
	if nfolds < 2:
		subsets = [
			pos_ids,  # Positive training set
			pos_ids,  # Positive test set
			neg_ids,  # Negative training set
			neg_ids  # Negative test set
		]
		# Printing fold number:
		print('fold {}'.format(1))
		train_and_evaluate(minsup, graph_database, subsets)

	# Otherwise: performs k-fold cross-validation:
	else:
		pos_fold_size = len(pos_ids) // nfolds
		neg_fold_size = len(neg_ids) // nfolds
		for i in range(nfolds):
			# Use fold as test set, the others as training set for each class;
			# identify all the subsets to be maintained by the graph mining algorithm.
			subsets = [
				numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
				pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
				numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
				neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
			]
			# Printing fold number:
			print('fold {}'.format(i+1))
			train_and_evaluate(k, minsup, graph_database, subsets)


def train_and_evaluate(k, minsup, database, subsets):
	task = ConfidencePositiveGraphs2(k, minsup, database, subsets)  # Creating task
	gSpan(task).run()  # Running gSpan
	# Creating feature matrices for training and testing:
	features = task.get_feature_matrices()
	train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
	train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
	test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
	test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

	classifier = tree.DecisionTreeClassifier(random_state=1)
	# classifier = naive_bayes.GaussianNB(random_state=1)  # Creating model object
	classifier.fit(train_fm, train_labels)  # Training model

	predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

	accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

	# Printing frequent patterns along with their positive support:

	
	keys = task.patterns.keys()
	# print(len(keys))
	for key in keys:
		for pattern, a in task.patterns[key]:
			confidence = key[0]
			support = key[1] # This will have to be replaced by the confidence and support on both classes
			print('{} {} {}'.format(pattern, confidence, support))
			# print(a)
	# printing classification results:
	print(predicted)
	print('accuracy: {}'.format(accuracy))
	print()  # Blank line to indicate end of fold.


def example3():
	a = 1

	if a == 1:
		args = sys.argv
		database_file_name_pos = args[1]  # First parameter: path to positive class file
		database_file_name_neg = args[2]  # Second parameter: path to negative class file
		k  = int(args[3])
		minsup = int(args[4])  # Third parameter: minimum support (note: this parameter will be k in case of top-k mining)
		nfolds = int(args[5])  # Fourth parameter: number of folds to use in the k-fold cross-validation.
	else:
		database_file_name_pos = 'data/molecules-small.pos'
		database_file_name_neg = 'data/molecules-small.neg'
		k = 5
		minsup = 5
		nfolds = 4

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	# print(graph_database._graphs[0].display())
	neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	# If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
	if nfolds < 2:
		subsets = [
			pos_ids,  # Positive training set
			pos_ids,  # Positive test set
			neg_ids,  # Negative training set
			neg_ids  # Negative test set
		]
		# Printing fold number:
		print('fold {}'.format(1))
		train_and_evaluate(minsup, graph_database, subsets)

	# Otherwise: performs k-fold cross-validation:
	else:
		pos_fold_size = len(pos_ids) // nfolds
		neg_fold_size = len(neg_ids) // nfolds
		for i in range(nfolds):
			# Use fold as test set, the others as training set for each class;
			# identify all the subsets to be maintained by the graph mining algorithm.
			subsets = [
				numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
				pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
				numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
				neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
			]
			# Printing fold number:
			print('fold {}'.format(i+1))
			Sequential_Covering(k, minsup, graph_database, subsets)


def Sequential_Covering(k, minsup, database, subsets):
	origin_label = copy.deepcopy([subsets[1], subsets[3]])
	new_subsets = []
	for subset in subsets:
		if type(subset) != type([]):
			new_subset = subset.tolist()
			new_subsets.append(new_subset)
		else:
			new_subsets.append(subset)
	pattern_dic = {}
	test_pred = {}
	for _ in range(k):
		task = ConfidencePositiveGraphs3(1, minsup, database, new_subsets)  # Creating task
		gSpan(task).run()  # Running gSpan
		new_pattern = task.patterns
		keys = new_pattern.keys()
		for key in keys:
			# print(key)
			pattern_list = new_pattern[key]
			if len(pattern_list) == 1:
				pattern = pattern_list[0]
			else:
				DFS_list = [pattern[0] for pattern in pattern_list]
				min_DFS = min(DFS_list)
				DFS_index = DFS_list.index(min_DFS)
				# get lowest
				pattern = pattern_list[DFS_index]
			# print(pattern[0], key)
			pattern_dic[pattern[0]] = (pattern[2],key)
			example_list = pattern[1]
			test_list = example_list[1] + example_list[3]
			# print(example_list)
			for item in test_list:
				test_pred[item] = pattern[2]
			new_subsets = RemoveX1FromX2(example_list, new_subsets)
	
	
	test_list = new_subsets[1] + new_subsets[3]
	# print(new_subsets)
	length_pos, length_neg = len(new_subsets[0]), len(new_subsets[2])
	if length_pos >= length_neg:
		default = 'pos'
	else:
		default = 'neg'
	for item in test_list:
			test_pred[item] = default
	# print('dic', pattern_dic)
	# print(test_pred)
	keys = test_pred.keys()
	key_list = [key for key in keys]
	key_list.sort()
	# print(key_list)
	test_prediction = [test_pred[key] for key in key_list]
	# print patterns
	keys = pattern_dic.keys()
	for key in keys:
		print('{} {} {}'.format(key, pattern_dic[key][1][0], pattern_dic[key][1][1]))
	# print prediction
	out_pred = []
	for pred in test_prediction:
		if pred == 'pos':
			out_pred.append(1)
		else:
			out_pred.append(-1)
	print(out_pred)
	# print accuracy
	keys = test_pred.keys()
	counter = 0
	# print(origin_label)
	for key in keys:
		if key in origin_label[0]:
			if test_pred[key] == 'pos':
				counter += 1
		if key in origin_label[1]:
			if test_pred[key] == 'neg':
				counter += 1
	accuracy = counter / len(keys)
	print('accuracy: {}'.format(accuracy))
	print()  # Blank line to indicate end of fold.

		


def RemoveX1FromX2(X1, X2):
	for index in range(len(X1)):
		sub1, sub2 = X1[index], X2[index]
		for item in sub1:
			sub2.remove(item)
	return X2


if __name__ == '__main__':
	a = 2
	
	
	if a == 1:
		example2()
	else:
		example3()
