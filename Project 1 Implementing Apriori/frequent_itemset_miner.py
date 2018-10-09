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


def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    # TODO: implementation of the apriori algorithm
    print("Not implemented")


def depth_first(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    # TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
    original_dataset = Dataset(filepath)
    database = original_dataset.transactions
    minFrequency = minFrequency * len(database)
    item_list = original_dataset.items
    operation_list = [item for item in item_list]
    database = vertical_representation(item_list, database)
    # print('database: ', database)
    for item in item_list:
        candidate_list = [item]
        depth_search(candidate_list, item, minFrequency, database)


def depth_search(candidate_list, item, minFrequency, database):
    """Undergoes depth first search"""
    operation_database = copy.deepcopy(database)
    support, projected = projected_database(item, operation_database)
    if support >= minFrequency:
        # if len(candidate_list) == 1:
        print('frequent item: ', candidate_list, 'fre: ', support)
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
            depth_search(copy_candidate, item, minFrequency, projected)


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





pwd = os.getcwd()
Dataset_Path = "Datasets"
Dataset_Name = "accidents.dat"
Dataset_Path = os.path.join(pwd, Dataset_Path, Dataset_Name)
# print(Dataset_Path)
depth_first(Dataset_Path, 0.9)

# lista = [1, [-1], [-1], 5, 4, 6]
# length = len(lista)
# for i in range(length):
#     if lista[length - 1 -i] == [-1]:
#         break
# newlist = [ lista[j + i] for j in range(length - i)]
# print(newlist)




