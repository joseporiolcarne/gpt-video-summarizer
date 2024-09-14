import unittest


from src.utils import sanitize_filename, is_url_accessible


class TestUtils(unittest.TestCase):

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename('test<>:"/\\|?*.txt'), "test.txt")
        self.assertEqual(sanitize_filename("  test  "), "test")
        self.assertEqual(sanitize_filename("test/file"), "testfile")

    def test_is_url_accessible(self):
        # This test assumes that the URL is accessible and belongs to the allowed domain
        self.assertTrue(
            is_url_accessible("https://www.youtube.com/watch?v=3k89FMJhZ00&t=52s")
        )

        # Test with an invalid URL format
        with self.assertRaises(ValueError):
            is_url_accessible("invalid-url")

        # Test with a URL not belonging to the allowed domain
        with self.assertRaises(ValueError):
            is_url_accessible("https://www.example.com")


if __name__ == "__main__":
    unittest.main()
