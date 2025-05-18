from src.visualisation import heatmap

import unittest

import numpy as np


class TestCreateHeatMapFromMatrix(unittest.TestCase):

    def test_chunk_matrix_wrong_value(self):
        self.assertRaises(ValueError, heatmap.chunk_matrix, [], 0)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[], [], []], 0)
        self.assertRaises(ValueError, heatmap.chunk_matrix, ["123", "345", "678"], 2)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [["1", "2", "3"], ["3", "4", "5"], ["6", "7", "-8"]], 2)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4, 5]], 2)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], "2")
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], [])
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], {})
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], [1, 2, 3])
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], 0)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4]], -1)
        self.assertRaises(ValueError, heatmap.chunk_matrix, [[1, 2], [2, 3], [3, 4, 5]], 2)
        
    
    def test_chunk_matrix_wrong_type(self):
        self.assertRaises(TypeError, heatmap.chunk_matrix, "2", 1)
        self.assertRaises(TypeError, heatmap.chunk_matrix, "2345", 1)
        self.assertRaises(TypeError, heatmap.chunk_matrix, 2, 5)
        self.assertRaises(TypeError, heatmap.chunk_matrix, {}, 5)

    
    def test_chunk_matrix_max_rows_less_than_rows_in_matrix(self):
        result = heatmap.chunk_matrix([[1, 2], [2, 3], [3, 4], [5, 6]], 2)
        expected = [np.asarray([[1, 2], [2, 3]]), np.asarray([[3, 4], [5, 6]])]
        for idx, chunk in enumerate(result):
            self.assertIsNone(np.testing.assert_array_equal(chunk, expected[idx]))

    
    def test_chunk_matrix_max_rows_more_than_rows_in_matrix(self):
        result = heatmap.chunk_matrix ([[1, 2], [2, 3], [3, 4]], 5)
        expected = [np.asarray([[1, 2], [2, 3], [3, 4]])]
        for idx, chunk in enumerate(result):
            self.assertIsNone(np.testing.assert_array_equal(chunk, expected[idx]))


if __name__ == '__main__':
    unittest.main()