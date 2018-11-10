import os
import pickle
import copy
import time 

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            all_trans = [line.strip() for line in open(filepath, 'r')]
            transaction = []
            for line in all_trans:
                if line:
                    transaction.append(line)
                else:
                    for item in transaction:
                        item = item.split(' ')[0]
                        self.items.add(item)
                    self.transactions.append(transaction)
                    transaction = []
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


class Node:
    def __init__(self, item, dataset, parent=None, children=None):
        self.item = item
        self.dataset = dataset
        self.parent = parent
        if self.parent == None:
            self.depth = 1
        else:
            self.parent.add_children([self])
            self.depth = self.parent.depth + 1
        if children == None:
            self.children = []
        else:
            self.children = [child for child in children]

    def add_children(self, children):
        self.children = self.children + children

    def get_parent(self):
        return self.parent

    def get_children(self):
        return [child for child in self.children]

    def get_all(self):
        parents = [self]
        node = self
        while node.parent:
            parents.append(node.parent)
            node = node.parent
        new_parents = []
        for index in range(len(parents)):
            new_parents.append(parents[-index - 1])
        return new_parents


### SPADE ###
def depth_first(item, items, parent):
    item_index = items.index(item)
    item_support = Unique_Index(parent.dataset[item_index])
    projected_dataset = Projeted(item_index, parent.dataset)
    new_node = Node(item, projected_dataset, parent)
    if new_node.depth == 2:
        print(new_node.item)
        print(new_node.parent.item)
        print('back')
        return 0
    for item in items:
        depth_first(item, items, new_node)


def Vertical_Representation(items, transactions):
    # items = list(items)
    v_items = [[] for _ in range(len(items))]
    for t_index in range(len(transactions)):
        transaction = transactions[t_index]
        for i_index in range(len(transaction)):
            v_items[items.index(transaction[i_index].split(' ')[0])].append([t_index + 1, i_index + 1])
    return v_items


def Projeted(op_item, dataset, minSupport=1):
    """Returns projected dataset"""
    unique = Unique_Index(dataset[op_item])
    ruler_i = [element[0] for element in unique]
    ruler_j = [element[1] for element in unique]
    new_dataset = copy.deepcopy(dataset)
    if len(unique) >= minSupport:
        for item_index in range(len(dataset)):
            if item_index == op_item:
                for element in dataset[op_item]:
                    if element in unique:
                        new_dataset[op_item].remove(element)
            else:
                current_data = new_dataset[item_index]
                if len(current_data) >= minSupport:
                    for element in dataset[item_index]:
                        if element[0] not in ruler_i:
                            current_data.remove(element)
                        elif element[1] < ruler_j[ruler_i.index(element[0])]:
                            current_data.remove(element)
    #                 if len(current_data) < minSupport:
    #                     current_data = []
    # else:
    #     new_dataset[op_item] = []
    return new_dataset


def Unique_Index(item_list):
    ruler = []
    result = []
    for element in item_list:
        i_index = element[0]
        if i_index not in ruler:
            ruler.append(i_index)
            result.append(element)
    return result


def SPADE(filepath, items, dataset):
    # data = Dataset(Final_Path)
    # items = data.items
    # items = list(items)
    # transactions = data.transactions
    # new = Vertical_Representation(items, transactions)
    Empty = Node('empty', dataset)
    for item in items:
        depth_first(item, items, Empty)
    print('ok')


### PrefixSpan ###
def Index_Transaction(items, transactions):
    """For each symbol in each transaction, maintain a list of its positions"""
    index_version = []
    for i in range(len(transactions)):
        new_index = []
        exist_item = []
        for j in range(len(transactions[i])):
            element = transactions[i][j]
            if element not in exist_item:
                exist_item.append(element)
                new_index.append([element, [j]])
            else:
                element_index = exist_item.index(element)
                new_index[element_index][1].append(j)
        index_version.append(new_index)
    return index_version


def PrefixSpan(filepath1, filepath2, k):
    """Initialize PrefixSpan Search"""
    ### Read files ###
    data1 = Dataset(filepath1)
    items1 = data1.items
    transactions1 = data1.transactions

    data2 = Dataset(filepath2)
    items2 = data2.items
    transactions2 = data2.transactions
    ### Combine datasets ###
    items = list(items1 | items2)
    transactions = transactions1 + transactions2

    for i in range(len(transactions)):
        for j in range(len(transactions[i])):
            transactions[i][j] = transactions[i][j].split(' ')[0]
    new_transactions = Index_Transaction(items, transactions)

    cursor = [-1 for _ in range(len(transactions))]
    Root = Node('root', cursor)
    MinFrequency = 0.9
    MinSupport = MinFrequency * len(transactions)
    # print(new_transactions[0])
    valid_list = []
    for item in items:
        Depth_First(item, items, new_transactions, Root, MinSupport, valid_list)
    # result_length = len(valid_list)

    number = Get_Support(valid_list)
    while number < k:
        MinSupport = MinSupport - (k - number)
        for node in valid_list:
            for item in items:
                Depth_First(item, items, new_transactions, node, MinSupport, valid_list)
        number = Get_Support(valid_list)

    for element in valid_list:
        result = []
        for node in element.get_all():
            result.append(node.item)
        print(result)

    


def Depth_First(item, items, dataset, parent, MinSupport, valid_list):
    """Given an item, find the position"""
    current_cursor = [-1 for _ in range(len(dataset))]
    parent_cursor = parent.dataset
    # print('item', item)
    # print('parent', parent_cursor)
    for index in range(len(dataset)):
        current_transaction = dataset[index]
        # print('trans', current_transaction)
        start_index = parent_cursor[index] + 1
        transaction_items = []
        for element in current_transaction:
            transaction_items.append(element[0])
        if item not in transaction_items:
            current_cursor[index] = 10000
        else:
            candidates = current_transaction[transaction_items.index(item)][1]
            if candidates[-1] < start_index:
                current_cursor[index] = 10000
            else:
                for order in candidates:
                    if order >= start_index:
                        current_cursor[index] = order
                        break
    # print('current', current_cursor)
    number = 0
    for element in current_cursor:
        if element != 10000:
            number += 1
    if number >= MinSupport:
        new_node = Node(item, current_cursor, parent)
        valid_list.append(new_node)

        # mylist = []
        # result = new_node.get_all()
        # for element in result:
        #     mylist.append(element.item)
        # print(mylist)

        for item in items:
            Depth_First(item, items, dataset, new_node, MinSupport, valid_list)


def Get_Support(node_list):
    overall_length = set()
    for element in node_list:
        result = element.get_all()
        overall_length.add(len(result))
    overall_length = list(overall_length)
    return len(overall_length)


pwd = os.getcwd()
Dataset_Path = "Datasets"
Subpath = "Test"
Dataset_Name1 = "positive.txt"
Dataset_Name2 = "negative.txt"
Final_Path1 = os.path.join(pwd, Dataset_Path, Subpath, Dataset_Name1)
Final_Path2 = os.path.join(pwd, Dataset_Path, Subpath, Dataset_Name2)

# SPADE(Final_Path, items, new)
PrefixSpan(Final_Path1, Final_Path2, 5)



my_trans = [
    [
        [1,1], [1,3], [2,1], [3,2], [3,3], [4,4]
    ],
    [
        [1,2], [2,3], [2,4], [3,1], [4,2], [4,3]
    ],
    [
        [1,4], [2,2], [3,4], [4,1]
    ]
]

items = ['A', 'B', 'C']
test_trans = [
    ['A', 'B', 'A', 'C'], ['B', 'C'], ['C', 'A']
]
# print(Index_Transaction(items, test_trans))


a = Node('a', 'a')
b = Node('b', 'b', a)
c = Node('c', 'c', b)
d = Node('c', 'c', c)

