"""
File name: test_similarity
Author: Fran Moreno
Last Updated: 11/6/2025
Version: 1.0
Description: TOFILL
"""
import unittest

from parameterized import parameterized

import clem.utils.similarity as module


class TestIsAMatch(unittest.TestCase):
    @parameterized.expand([
        # Almost equal strings
        ("000", 'apple', 'apple', True),
        ("001", 'banana', 'banan', True),
        ("002", 'cherry', 'chery', True),
        ("003", 'date', 'dte', True),
        ("004", 'elephant', 'elepant', True),
        ("005", 'fish', 'fihs', False),
        ("006", 'grape', 'grap', True),
        ("007", 'honey', 'honeyy', True),
        ("008", 'ice', 'icee', True),
        ("009", 'jackfruit', 'jackfuit', True),

        # Containment: one string inside the other
        ("010", 'adobe', 'adobe acrobat engine', False),
        ("011", 'acrobat', 'adobe acrobat system', False),
        ("012", 'system', 'adobe acrobat system', False),
        ("013", 'engine', 'adobe acrobat engine', False),
        ("014", 'val', 'val1', True),
        ("015", 'val', 'val123', True),
        ("016", 'core', 'core i7 processor', False),
        ("017", 'i7', 'core i7 processor', False),
        ("018", 'processor', 'core i7 processor', True),
        ("019", 'python', 'python 3.10', True),

        # Long vs short strings
        ('020', 'adobe acrobat engine', 'acrobat', False),
        ('021', 'adobe acrobat system', 'system', False),
        ('022', 'core i7 processor', 'core', False),
        ('023', 'machine learning algorithm', 'learning', False),
        ('024', 'neural network', 'network', True),
        ('025', 'data processing unit', 'unit', False),
        ('026', 'high performance computing', 'high', False),
        ('027', 'deep learning', 'learning', True),
        ('028', 'convolutional neural network', 'network', False),
        ('029', 'recurrent neural network', 'recurrent', False),

        # Small edits / typos
        ('030', 'color', 'colour', True),
        ('031', 'theatre', 'theater', True),
        ('032', 'organize', 'organise', True),
        ('033', 'center', 'centre', True),
        ('034', 'analyze', 'analyse', True),
        ('035', 'realize', 'realise', True),
        ('036', 'travelling', 'traveling', True),
        ('037', 'program', 'prrogram', True),
        ('038', 'function', 'functon', True),
        ('039', 'variable', 'varible', True),

        # Completely different strings
        ('040', 'apple', 'orange', False),
        ('041', 'cat', 'dog', False),
        ('042', 'house', 'car', False),
        ('043', 'table', 'chair', False),
        ('044', 'sun', 'moon', False),
        ('045', 'water', 'fire', False),
        ('046', 'earth', 'mars', False),
        ('047', 'python', 'java', False),
        ('048', 'linux', 'windows', False),
        ('049', 'google', 'microsoft', False),

        # Single-character differences (really short strings)
        ('050', 'abc', 'abd', False),
        ('051', 'abcd', 'abce', False),
        ('052', 'xyz', 'xya', False),
        ('053', 'mnop', 'mnopq', True),
        ('054', 'test', 'tast', False),
        ('055', 'fast', 'fist', False),
        ('056', 'last', 'lust', False),
        ('057', 'mast', 'mask', False),
        ('058', 'desk', 'desq', False),
        ('059', 'task', 'tusk', False),

        # Single-character differences (short strings)
        ('060', 'abcdef', 'abcdeg', True),
        ('061', 'ghijkl', 'ghijxl', True),
        ('062', 'mnopqr', 'mnoppr', True),
        ('063', 'stuvwx', 'stuvwx', True),
        ('064', 'yzabcd', 'yzabce', True),
        ('065', 'efghij', 'efghik', True),
        ('066', 'klmnop', 'klmnqp', True),
        ('067', 'qrstuv', 'qrstuv', True),
        ('068', 'wxyzab', 'wxyzac', True),
        ('069', 'cdefgh', 'cdefgi', True),

        # Single-character differences (medium strings)
        ('070', 'abcdefghij', 'abcdefghim', True),
        ('071', 'klmnopqrst', 'klmnopqrsu', True),
        ('072', 'uvwxyzabcd', 'uvwxyzabce', True),
        ('073', 'efghijklmn', 'efghijklmz', True),
        ('074', 'opqrstuvwx', 'opqrstuvwy', True),
        ('075', 'yzabcdefgh', 'yzabcdefgi', True),
        ('076', 'ijklmnopqr', 'ijklmnopqs', True),
        ('077', 'stuvwxyzab', 'stuvwxyzac', True),
        ('078', 'cdefghijkl', 'cdefghijkm', True),
        ('079', 'mnopqrstuv', 'mnopqrstvz', True),

        # Case differences
        ('080', 'Python', 'python', True),
        ('081', 'JAVA', 'java', True),
        ('082', 'Linux', 'linux', True),
        ('083', 'Windows', 'windows', True),
        ('084', 'Apple', 'apple', True),
        ('085', 'Microsoft', 'microsoft', True),
        ('086', 'GitHub', 'github', True),
        ('087', 'Docker', 'docker', True),
        ('088', 'Kubernetes', 'kubernetes', True),
        ('089', 'TensorFlow', 'tensorflow', True),

        # Strings with numbers
        ('090', 'val1', 'val2', False),
        ('091', 'v123', 'v124', False),
        ('092', 'version1', 'version2', True),
        ('093', 'item001', 'item002', True),
        ('094', 'product10', 'product11', True),
        ('095', 'batch5', 'batch6', True),
        ('096', 'sample123', 'sample124', True),
        ('097', 'release1.0', 'release1.1', True),
        ('098', 'update2023', 'update2024', True),
        ('099', 'build42', 'build43', True),

        # Mixed letters and numbers
        ('100', 'abc123', 'abc124', True),
        ('101', 'x9y8', 'x9y9', False),
        ('102', 'test007', 'test008', True),
        ('103', 'v2alpha', 'v2beta', False),
        ('104', 'node12', 'node13', True),
        ('105', 'itemA1', 'itemA2', True),
        ('106', 'keyX9', 'keyY9', True),
        ('107', 'code100', 'code101', True),
        ('108', 'task42', 'task43', True),
        ('109', 'job007', 'job008', True),

        # Substring repetition / partial overlap
        ('110', 'abcabc', 'abc', True),
        ('111', '123123', '123', True),
        ('112', 'hellohello', 'hello', True),
        ('113', 'testtest', 'test', True),
        ('114', 'xyxyxy', 'xy', False),
        ('115', 'foobarfoo', 'foo', False),
        ('116', 'barbar', 'bar', True),
        ('117', 'pingpong', 'ping', True),
        ('118', 'repeatrepeat', 'repeat', True),
        ('119', 'lolol', 'lol', True),

        #'000',  Special / miscellaneous cases
        ('120', 'a', 'a', True),
        ('121', 'ab', 'ba', False),
        ('122', 'abc', 'cab', False),
        ('123', 'abcd', 'dcba', False),
        ('124', '1234', '4321', False),
        ('125', 'python3.10', 'python3.9', True),
        ('126', 'tensorflow2', 'tensorflow1', True),
        ('127', 'keras', 'keras', True),
        ('128', 'pytorch', 'torch', True),
    ])
    def test_cases(self, _test_idx, string1, string2, expected):
        actual = module.is_a_match(string1, string2)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()