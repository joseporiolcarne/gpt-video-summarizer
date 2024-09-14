import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from io import StringIO


# Get the absolute path of the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.main import main


class TestMain(unittest.TestCase):

    @patch("src.main.is_url_accessible", return_value=True)
    @patch("src.main.YouTube")
    @patch("src.main.transcribe_audio", return_value="Sample transcript")
    @patch("src.main.read_srt", return_value="Sample SRT content")
    @patch("src.main.summarize", return_value="Sample summary")
    @patch("src.main.sanitize_filename", return_value="sample_video_title")
    def test_main(
        self,
        mock_sanitize_filename,
        mock_summarize,
        mock_read_srt,
        mock_transcribe_audio,
        mock_youtube,
        mock_is_url_accessible,
    ):
        test_args = ["main.py", "https://www.youtube.com/watch?v=3k89FMJhZ00&t=52s"]
        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                main()
                self.assertIn("Summary copied to clipboard.", fake_out.getvalue())
                self.assertIn("Successfully saved summary as", fake_out.getvalue())


if __name__ == "__main__":
    unittest.main()
