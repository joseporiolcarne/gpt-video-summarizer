import re
import requests
from requests.exceptions import RequestException
import validators
from urllib.parse import urlparse


def sanitize_filename(filename):
    """
    Sanitize the provided filename by removing invalid characters and trailing spaces.
    """
    # Windows
    # Remove characters not allowed in Windows file names
    windows_invalid_chars = r'[<>:"/\\|?*]'
    filename = re.sub(windows_invalid_chars, "", filename)

    # Remove trailing spaces and periods
    filename = filename.strip(" .")

    # Linux
    # Replace forward slashes with underscores
    filename = filename.replace("/", "_")

    return filename


def is_url_accessible(url, allowed_domain="youtube.com"):
    """
    Check if the provided URL is valid, belongs to the specified domain, and is accessible.

    :param url: The URL to check.
    :param allowed_domain: The domain that the URL must belong to (default: 'youtube.com').
    :return: True if the URL is valid, belongs to the allowed domain, and is accessible. Raises ValueError otherwise.
    """
    # Validate URL format
    if not validators.url(url):
        raise ValueError("Invalid URL format.")

    # Parse the URL to extract the domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Check if the domain matches the allowed domain
    if allowed_domain not in domain:
        raise ValueError(
            f"URL must belong to the domain: {allowed_domain}. Found: {domain}"
        )

    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=5)
        # Check if the status code indicates success
        if response.status_code == 200:
            return True
        else:
            print(f"URL is not accessible. Status code: {response.status_code}")
            return False
    except RequestException as e:
        print(f"Error accessing URL: {e}")
        return False
