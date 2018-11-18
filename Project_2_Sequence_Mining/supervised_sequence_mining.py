import os
import pickle
import copy
import time
import sys


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
        if self.parent is None:
            self.depth = 1
        else:
            self.parent.add_children([self])
            self.depth = self.parent.depth + 1
        if children is None:
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


def Weighted_Relative_ccuracy(P, N, p, n):
    """ Returns Weighted Relative ccuracy """
    coefficient = (P / (P + N)) * (N / (P + N))
    answer = coefficient * (p / P - n / N)
    return round(answer, 5)


def supervised_sequence_mining(filepath1, filepath2, k):
    """ Initialize Supervised Sequence Mining """
    # Read files
    data1 = Dataset(filepath1)
    items1 = data1.items
    transactions1 = data1.transactions
    data2 = Dataset(filepath2)
    items2 = data2.items
    transactions2 = data2.transactions
    # Combine datasets
    items = list(items1 | items2)
    trans_list = [transactions1, transactions2]
    # Preprocess datasets
    new_trans_list = []
    for transactions in trans_list:
        for i in range(len(transactions)):
            for j in range(len(transactions[i])):
                transactions[i][j] = transactions[i][j].split(' ')[0]
        new_trans_list.append(Index_Transaction(items, transactions))
    # Initialize Root
    cursor = []
    for index in range(len(new_trans_list)):
        current_transaction = new_trans_list[index]
        cursor.append([-1 for _ in range(len(current_transaction))])
    Root = Node('root', cursor)
    # Start Search
    valid_list = {}
    lower_bound = 0
    min_score = 1
    parameter = [k]
    for item in items:
        Depth_First(item, items, new_trans_list, Root, parameter, valid_list)
    # Output All Frequent Sequence #
    # print(len(valid_list.keys()))
    result = []
    for key in valid_list.keys():
        result = result + valid_list[key]
    all_sequence = All_Frequent_Sequence(result)
    output_sequence = []
    for sequence in all_sequence:
        output = [element[0] for element in sequence]
        output_sequence.append(output)
        new_output = ''
        for element in output:
            new_output = new_output + element + ', '
        new_output = new_output[: -2]
        support_p = sequence[-1][1][0]
        support_n = sequence[-1][1][1]
        score = sequence[-1][2]
        print('[{}]'.format(new_output), support_p,
              support_n, score)


def Depth_First(item, items, new_trans_list, parent, parameter, valid_list):
    """Given an item, find the position"""
    [k] = parameter
    new_cursor = []
    support = []
    for index in range(len(new_trans_list)):
        dataset = new_trans_list[index]
        current_cursor = copy.deepcopy(parent.dataset[index])
        parent_cursor = parent.dataset[index]
        number = 0
        new_items = set()
        for index in range(len(dataset)):
            flag = 0
            if parent_cursor[index] != 10000:
                current_transaction = dataset[index]
                start_index = parent_cursor[index] + 1
                for element in current_transaction:
                    if item == element[0]:
                        flag = 1
                        candidates = element[1]
                        if candidates[-1] < start_index:
                            current_cursor[index] = 10000
                        else:
                            for order in candidates:
                                if order >= start_index:
                                    current_cursor[index] = order
                                    number += 1
                                    for element in current_transaction:
                                        new_items.add(element[0])
                                    break
                        break
                if flag == 0:
                    current_cursor[index] = 10000
        new_cursor.append(current_cursor)
        support.append(number)
    if support == [0, 0]:
        return 0
    # Create new node
    P = len(new_trans_list[0])
    N = len(new_trans_list[1])
    coefficient = (P / (P + N)) * (N / (P + N))
    score = Weighted_Relative_ccuracy(P, N,
                                      support[0], support[1])
    new_node = Node([item, support, score], new_cursor, parent)
    # print(new_node.item)
    # print(new_node.dataset)
    key = score
    score_list = valid_list.keys()
    # print(score_list)
    if len(score_list) == 0:
        min_score = -1
    else:
        min_score = min(score_list)
    lower_bound = P * min_score / coefficient
    if support[0] < lower_bound:
        return 0
    if len(valid_list) < k:
        try:
            valid_list[key].append(new_node)
        except:
            valid_list[key] = [new_node]
        # Start new search
        for item in items:
            Depth_First(item, items, new_trans_list,
                        new_node, parameter, valid_list)
    else:
        if score >= min_score:
            try:
                valid_list[key].append(new_node)
            except:
                valid_list[key] = [new_node]
            if len(valid_list) > k:
                valid_list.pop(min_score)
            # Start new search
            for item in items:
                Depth_First(item, items, new_trans_list,
                            new_node, parameter, valid_list)
        elif support[0] >= lower_bound:
            # Start new search
            for item in items:
                Depth_First(item, items, new_trans_list,
                            new_node, parameter, valid_list)


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


def All_Frequent_Sequence(node_list):
    all_sequence = []
    for element in node_list:
        result = []
        for node in element.get_all():
            result.append(node.item)
        result.remove(result[0])
        all_sequence.append(result)
    return all_sequence


def main():
    a = 1
    if a == 1:
        pos_filepath = sys.argv[1]  # filepath to positive class file
        neg_filepath = sys.argv[2]  # filepath to negative class file
        k = int(sys.argv[3])
        supervised_sequence_mining(pos_filepath, neg_filepath, k)
    else:
        pwd = os.getcwd()
        Dataset_Path = "Datasets"
        # Subpath = "Reuters"
        # Dataset_Name1 = "acq.txt"
        # Dataset_Name2 = "earn.txt"
        Subpath = "Test"
        Dataset_Name1 = "positive.txt"
        Dataset_Name2 = "negative.txt"
        Final_Path1 = os.path.join(pwd, Dataset_Path, Subpath, Dataset_Name1)
        Final_Path2 = os.path.join(pwd, Dataset_Path, Subpath, Dataset_Name2)
        supervised_sequence_mining(Final_Path1, Final_Path2, 6)


if __name__ == "__main__":
    main()
