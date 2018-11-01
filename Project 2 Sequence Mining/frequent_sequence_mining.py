import os
import pickle
import copy

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



pwd = os.getcwd()
Dataset_Path = "Datasets"
Subpath = "Reuters"
Dataset_Name = "acq.txt"
Final_Path = os.path.join(pwd, Dataset_Path, Subpath, Dataset_Name)
# data = Dataset(Final_Path)
# items = data.items
# items = list(items)
# transactions = data.transactions


# with open('transactions.pkl', 'wb') as f:
#     pickle.dump(transactions, f)
with open('item.pkl', 'rb') as f:
    items = pickle.load(f)
with open('transactions.pkl', 'rb') as f:
    transactions = pickle.load(f)

# with open('data.pkl', 'rb') as f:
#     new = pickle.load(f)
# SPADE(Final_Path, items, new)



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
# a = Projeted(1, my_trans, 1)
# print(a)

a = Node('a', 'a')
b = Node('b', 'b', a)
c = Node('c', 'c', b)
# print(c.depth)