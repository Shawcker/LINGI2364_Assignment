"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<write here your group, first name(s) and last name(s)>"
"""

import os
import copy

class Trie:
	class Node:
		def __init__(self, char):
			self.char = char
			self.childs = []
			self.nb = 1 #how many words are using this node
			self.last = False

	def __init__(self):
	    self.root = self.Node("*")

	def insert(self, currentNode, word):
		rest = ""
		if len(word) == 0: #no char to insert (the previous node inserted this node's character, so the "last" flag is valid)
			currentNode.last = True
			return
		else:
			letter = word[0]
			if len(word) != 1:
				rest = word[1:]
		foundChild = None
		for child in currentNode.childs:
			if child.char == letter: #if already exists in this node's children
				child.nb += 1
				foundChild = child
				break
		if foundChild is None: #create a new child node
			newChild = self.Node(letter)
			currentNode.childs.append(newChild)
			foundChild = newChild
		self.insert(foundChild, rest)

	def nbOccurences(self, word, currentNode):
		count = 0
		for node in currentNode.childs:
			if len(word) > 1:
				if node.char == word[0]:
					count += self.nbOccurences(word[1:], node)
			else: #last char of word
				if node.char == word[0]: 
					count += node.nb
			if node.char != word[0]:
				count += self.nbOccurences(word, node)
		return count


	def isPresent(self, currentNode, word):
		rest = ""
		if len(word) == 0:
			if currentNode.last == True:
				return True
			return False
		else:
			letter = word[0]
			if len(word) != 1:
				rest = word[1:]
		for child in currentNode.childs:
			if child.char == letter:
				return self.isPresent(child, rest) #todo: make it tail-recursive (with boolean if found)
		return False

	def countPrefix(self, currentNode, word): #nb of times the prefix is shared:
		#for instance: "she sells sea shells by the shore" => count("sh") is 3 and count("s") is 5
		rest = ""
		if len(word) == 0: 
			return currentNode.nb
		else:
			letter = word[0]
			if len(word) != 1:
				rest = word[1:]
		for child in currentNode.childs:
			if child.char == letter:
				return self.countPrefix(child, rest)
		return 0

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
                transaction = list(map(int, line.split(" ")))
                self.transactions.append(transaction)
                for item in transaction:
                    self.items.add(item)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self.transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]

# alternative_miner
def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    original_dataset = Dataset(filepath)
    database = original_dataset.transactions
    total_number = len(database)
    minFrequency = minFrequency * total_number
    item_list = original_dataset.items
    operation_list = [item for item in item_list]
    database = vertical_representation(item_list, database)
    # print('database: ', database)
    for item in item_list:
        candidate_list = [item]
        depth_search(candidate_list, item, minFrequency, total_number, database)


def depth_search(candidate_list, item, minFrequency,  total_number, database):
    """Undergoes depth first search"""
    operation_database = copy.deepcopy(database)
    support, projected = projected_database(item, operation_database)
    if support >= minFrequency:
        print('{} ({})'.format(candidate_list, support / total_number))
        # print('frequent item: ', candidate_list, 'fre: ', support / total_number)
        # print('projected: ', projected)
        # Create new item list
        length = len(projected)
        for index_1 in range(length):
            if projected[length - 1 - index_1] == [-1]:
                break
        item_list = [ length - index_1 + index_2 for index_2 in range(index_1)]
        # print('new', item_list)
        # Iterate through the list
        for item in item_list:
            copy_candidate = copy.deepcopy(candidate_list)
            copy_candidate.append(item)
            depth_search(copy_candidate, item, minFrequency, total_number, projected)


def projected_database(item, database):
    """Returns projected database"""
    removed_item = database[item - 1]
    # database.remove(removed_item)
    database[item - 1] = [-1]
    for i in range(len(database)):
        # print(database[i])
        if database[i] != [-1]:
            database[i] = list_intersections(removed_item, database[i])
    return len(removed_item), database


def list_intersections(list1, list2):
    """Returns intersections of two given lists"""
    len1 = len(list1)
    len2 = len(list2)
    i = j = 0
    intersections = []
    while (i < len1) and (j < len2):
        if list1[i] == list2[j]:
            intersections.append(list1[i])
            i += 1
            j += 1
        elif list1[i] > list2[j]:
            j += 1
        else:
            i += 1
    return intersections


def vertical_representation(item_list, database):
    """Returns vertical form of database"""
    vert_item = [[] for i in range(len(item_list))]
    for transaction_id in range(len(database)):
        transaction = database[transaction_id]
        for item in transaction:
            vert_item[item - 1].append(transaction_id + 1)
    return vert_item


# apriori
def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	tree = Trie()
	datas = Dataset(filepath)
	N = datas.trans_num() #number of transactions, used to compute the frequency of a set
	for transNb in range(0, len(datas.transactions)): 
		#sorting the elements in the transactions and insert in trie
		datas.transactions[transNb] = sortTransaction(datas.transactions[transNb])
		tree.insert(tree.root, datas.transactions[transNb])
	results = []
	layer1 = []
	for i in range(1, datas.items_num()+1):
			layer1.append([[i], 0])
	results.append(layer1)
	frequencyCounter(results, 1, minFrequency, tree, N)
	results[0] = prune(results, minFrequency)
	layer = 2
	while layer < (datas.items_num()+1):
		makeLayer(results, layer, minFrequency, N) #does the conbinations in an array, and adds it to results
		frequencyCounter(results, layer, minFrequency, tree, N)
		layer += 1
	toString(results, minFrequency)

	
def sortTransaction(transaction):
	#sort the transaction in order to be inserted in trie. Insertion sort 
	# because most transactions should be small in practice
	toSort = transaction
	for i in range(1, len(toSort)):
		val = toSort[i] 
		j = i-1
		while (j >= 0) and (val < toSort[j]): 
			toSort[j+1] = toSort[j] 
			j -= 1
		toSort[j+1] = val
	return toSort


def frequencyCounter(results, layer, minFrequency, tree, N):
	toDelete = []
	layerIdx = layer - 1
	for i in range(0, len(results[layerIdx])):
		subset = results[layerIdx][i][0]
		support = tree.nbOccurences(subset, tree.root)
		freq = float(support)/float(N)
		results[layerIdx][i][1] = freq
		if freq < minFrequency:
			toDelete.append(i)
	i = 0
	while i < len(toDelete):
		del results[layerIdx][toDelete[i]-i]
		i+=1


def makeCombination(itemL1, prevLayerSet, layerNum, layerN):
	newSet = [prevLayerSet[i] for i in range(0, len(prevLayerSet))]
	newSet.append(itemL1)
	sortTransaction(newSet)
	for i in range(0, len(newSet)-1):
		if newSet[i] == newSet[i+1]:
			return None
	for i in range(0, len(layerN)):
		if layerN[i][0] == newSet:
			return None
	return newSet


def makeLayer(results, layer, minFrequency, N): #N is the total number of transactions
	layer1 = results[0]
	prevLayerIdx = layer - 2
	layerN = []
	for i in range(0, len(results[prevLayerIdx])):
		PrevFrequency = float(results[prevLayerIdx][i][1])
		if PrevFrequency >= minFrequency:
			for u in range(0, len(layer1)):
				candidate = makeCombination(layer1[u][0][0], results[prevLayerIdx][i][0], layer, layerN)
				if candidate is not None:
					layerN.append([candidate, 0])
	results.append(layerN)
	return


def prune(results, minFreq):
	arr= []
	for i in range(len(results)):
		for j in range(len(results[i])):
			if float(results[i][j][1]) >= float(minFreq):
				arr.append([results[i][j][0], float(results[i][j][1])])
	return arr


def toString(results, freq):
	# [<item 1>, <item 2>, ... <item k>] (<frequency>)
	outString = "["
	for layer in range(0, len(results)):
		for i in range(0, len(results[layer])):
			outString += str(results[layer][i][0]) +", "
	outString = outString[:-2]
	outString += "]"
	outString += " (" + str(freq) + ")"
	print(outString)



	



# pwd = os.getcwd()
# Dataset_Path = "Datasets"
# Dataset_Name = "chess.dat"
# Dataset_Path = os.path.join(pwd, Dataset_Path, Dataset_Name)
# # print(Dataset_Path)
# alternative_miner(Dataset_Path, 0.98)
# apriori(Dataset_Path, 0.98)

# lista = [1, [-1], [-1], 5, 4, 6]
# length = len(lista)
# for i in range(length):
#     if lista[length - 1 -i] == [-1]:
#         break
# newlist = [ lista[j + i] for j in range(length - i)]
# print(newlist)




