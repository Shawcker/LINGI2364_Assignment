import unittest
from frequent_itemset_miner import list_intersections

class TestLI(unittest.TestCase):
    
    def stUP(self):
        pass

    def test_list_intersection(self):
        assert list_intersections([1, 2], [2, 3]) == [2]


if __name__ == '__main__':
    unittest.main()
