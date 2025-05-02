import unittest

from src.data_preprocessing.matrix_transformation import equalize_row_length_in_sentence_matrix, batch_sentence_to_threshold

class TestEqualizeRowLengthInSentenceMatrix(unittest.TestCase):

    def test_invalid_type_cases(self):
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, '', 3)
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, 0, 3)
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, {}, 3)
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, {'a': 'b', 'a1': 0}, 3)

    def test_invalid_array_cases(self):
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, [], 3)
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, [[], []], 3)
        self.assertRaises(ValueError, equalize_row_length_in_sentence_matrix, [[0, 1, 2], []], 3)

    def test_valid_cases(self):
        self.assertEqual(equalize_row_length_in_sentence_matrix([[0, 1, 2], [0], [0], [0]], 3), [[0, 1, 2], [0, -3, -3], [0, -3, -3], [0, -3, -3]])
        self.assertEqual(equalize_row_length_in_sentence_matrix([[0, 1, 5, 9, 6, 7, 2], [0], [0, 1], [0, 1], [0, 1]], 3), [[0, 1, 5, 9, 6, 7, 2], [0, -3, -3, -3, -3, -3, -3], [0, 1, -3, -3, -3, -3, -3], [0, 1, -3, -3, -3, -3, -3], [0, 1, -3, -3, -3, -3, -3]])

class TestBatchSentence(unittest.TestCase):
    def test_invalid_type_cases(self):
        self.assertRaises(ValueError, batch_sentence_to_threshold, '', 3, 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, 0, 3, 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, {}, 3, 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, [], 3, 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, {'a': 'b', 'a1': 0}, 3, 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, [[0, 1, 2], [0], [0], [0]], '', 3)
        self.assertRaises(ValueError, batch_sentence_to_threshold, [[0, 1, 2], [0], [0], [0]], 3, [])

    def test_valid_cases(self):
        self.assertEqual(batch_sentence_to_threshold([0, 1, 2], 3, 3), [[0, 1, 2]])
        self.assertEqual(batch_sentence_to_threshold([1], 2, 3), [[1, -3]])
        self.assertEqual(batch_sentence_to_threshold([0, 1, 2], 2, 3), [[0, 1], [1, 2]])
        self.assertEqual(batch_sentence_to_threshold([0, 1, 2], 3, 3), [[0, 1, 2]])

if __name__ == '__main__':
    unittest.main()