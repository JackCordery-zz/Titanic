from main import main
import unittest


class TestMain(unittest.TestCase):

    def test(self):
        result = main()
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
