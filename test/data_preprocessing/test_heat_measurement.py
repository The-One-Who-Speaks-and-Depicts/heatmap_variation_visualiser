import unittest

from src.data_preprocessing.heat_measurement import transform_gec_like_to_heat, transform_binary_tagged_sequence_to_underscore_and_X



class TestTransformToSequence(unittest.TestCase):

    def test_transform_gec_like_to_heat(self):
        self.assertRaises(ValueError, transform_gec_like_to_heat, '')
        self.assertRaises(ValueError, transform_gec_like_to_heat, 1)
        self.assertRaises(ValueError, transform_gec_like_to_heat, ["a", "b"])
        self.assertRaises(ValueError, transform_gec_like_to_heat, {"a": "b"})

        self.assertEqual(transform_gec_like_to_heat("Як же {автору=>авторові:::error_type=G/Case} вдалося передбачити майбутнє?", [r'\{([^=>]+)=>([^:]+):::(variation|error)_type=(G/[^}]+|Spelling)\}']),"______XXXXXX______________________________")
        self.assertEqual(transform_gec_like_to_heat("це менш природний процес, {ніж=>аніж:::error_type=Spelling} споглядання", [r'\{([^=>]+)=>([^:]+):::(variation|error)_type=(G/[^}]+|Spelling)\}']), "__________________________XXX____________")
        self.assertEqual(transform_gec_like_to_heat("тренду{=>;:::error_type=Punctuation}", [r'\{([^=>]+)=>([^:]+):::(variation|error)_type=(G/[^}]+|Spelling)\}']), "______")
        self.assertEqual(transform_gec_like_to_heat("читання {–=>—:::error_type=Punctuation} це менш природний процес", [r'\{([^=>]+)=>([^:]+):::(variation|error)_type=(G/[^}]+|Spelling)\}']), "__________________________________")

    def test_trasform_binary_tagged_sequence_to_underscore_and_X(self):
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, '')
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, 1)
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, ["a", "b"])
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, {"a": "b"})
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, [])
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, [("a", 1), ("b", "c")])
        self.assertRaises(ValueError, transform_binary_tagged_sequence_to_underscore_and_X, [("a", "1"), ("b", 0)])

        self.assertEqual(transform_binary_tagged_sequence_to_underscore_and_X([("aaa", 1), ("bb", 0)]), "XXX___")
        self.assertEqual(transform_binary_tagged_sequence_to_underscore_and_X([("aaa", 1)]), "XXX")
        self.assertEqual(transform_binary_tagged_sequence_to_underscore_and_X([("aaa", 0)]), "___")
        self.assertEqual(transform_binary_tagged_sequence_to_underscore_and_X([("наприклад", 0), (",", 0), ("проект", 0), ("«", 0), ("Життєлюб", 3), ("»", 0), ("Гаріка", 1), ("Корогодського", 2), (".", 0) ]), "_____________________XXXXXXXX___XXXXXX_XXXXXXXXXXXXX__")



if __name__ == '__main__':
    unittest.main()
