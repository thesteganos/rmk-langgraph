import unittest
from unittest.mock import patch, mock_open
import os
import sys

# Add the parent directory to sys.path to allow imports from knowledge_pipeline
# This assumes the tests are run from the repository root or that the 'src' or parent dir is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_pipeline import get_processed_urls, log_processed_url, generate_full_search_url_and_base # Add more imports as needed

# Constants from knowledge_pipeline that might be needed for mocking
PROCESSED_URLS_LOG = "processed_urls.log"

class TestKnowledgePipelineIO(unittest.TestCase):

    @patch('os.path.exists')
    def test_get_processed_urls_file_not_exists(self, mock_exists):
        mock_exists.return_value = False
        self.assertEqual(get_processed_urls(), set())
        mock_exists.assert_called_once_with(PROCESSED_URLS_LOG)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="")
    def test_get_processed_urls_empty_file(self, mock_file_open, mock_exists):
        mock_exists.return_value = True
        self.assertEqual(get_processed_urls(), set())
        mock_exists.assert_called_once_with(PROCESSED_URLS_LOG)
        mock_file_open.assert_called_once_with(PROCESSED_URLS_LOG, "r")

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="http://example.com/page1\nhttp://example.com/page2\n")
    def test_get_processed_urls_with_data(self, mock_file_open, mock_exists):
        mock_exists.return_value = True
        expected_urls = {"http://example.com/page1", "http://example.com/page2"}
        self.assertEqual(get_processed_urls(), expected_urls)
        mock_exists.assert_called_once_with(PROCESSED_URLS_LOG)
        mock_file_open.assert_called_once_with(PROCESSED_URLS_LOG, "r")

    @patch('os.path.exists')
    @patch('builtins.open', side_effect=IOError("Test IOError"))
    def test_get_processed_urls_io_error(self, mock_file_open, mock_exists):
        mock_exists.return_value = True
        with self.assertRaises(IOError):
            get_processed_urls()
        mock_exists.assert_called_once_with(PROCESSED_URLS_LOG)
        mock_file_open.assert_called_once_with(PROCESSED_URLS_LOG, "r")

    @patch('builtins.open', new_callable=mock_open)
    def test_log_processed_url_success(self, mock_file_open):
        test_url = "http://example.com/new_page"
        log_processed_url(test_url)
        mock_file_open.assert_called_once_with(PROCESSED_URLS_LOG, "a")
        mock_file_open().write.assert_called_once_with(test_url + "\n")

    @patch('builtins.open', side_effect=IOError("Test IOError on write"))
    def test_log_processed_url_io_error(self, mock_file_open):
        test_url = "http://example.com/another_page"
        with self.assertRaises(IOError):
            log_processed_url(test_url)
        mock_file_open.assert_called_once_with(PROCESSED_URLS_LOG, "a")

# ... (keep existing TestKnowledgePipelineIO class) ...

class TestURLGeneration(unittest.TestCase):

    def test_generate_full_search_url_and_base_simple(self):
        search_template = "https://www.example.com/search?q={query}"
        query = "test query"
        expected_full_url = "https://www.example.com/search?q=test+query"
        expected_base_url = "https://www.example.com"

        full_url, base_url = generate_full_search_url_and_base(search_template, query)
        self.assertEqual(full_url, expected_full_url)
        self.assertEqual(base_url, expected_base_url)

    def test_generate_full_search_url_and_base_with_path(self):
        search_template = "https://sub.example.org/some/path/results?search={query}&page=1"
        query = "another & test"
        expected_full_url = "https://sub.example.org/some/path/results?search=another+%26+test&page=1"
        expected_base_url = "https://sub.example.org" # Base URL should be scheme + netloc

        full_url, base_url = generate_full_search_url_and_base(search_template, query)
        self.assertEqual(full_url, expected_full_url)
        self.assertEqual(base_url, expected_base_url)

    def test_generate_full_search_url_and_base_no_query_in_path(self):
        search_template = "https://www.testsite.com/s?k={query}" # Common pattern for some sites
        query = "special chars: /?"
        expected_full_url = "https://www.testsite.com/s?k=special+chars%3A+%2F%3F"
        expected_base_url = "https://www.testsite.com"

        full_url, base_url = generate_full_search_url_and_base(search_template, query)
        self.assertEqual(full_url, expected_full_url)
        self.assertEqual(base_url, expected_base_url)

    def test_generate_full_search_url_and_base_http(self):
        search_template = "http://old.example.com/search?term={query}"
        query = "plain http"
        expected_full_url = "http://old.example.com/search?term=plain+http"
        expected_base_url = "http://old.example.com"

        full_url, base_url = generate_full_search_url_and_base(search_template, query)
        self.assertEqual(full_url, expected_full_url)
        self.assertEqual(base_url, expected_base_url)

if __name__ == '__main__':
    unittest.main()
