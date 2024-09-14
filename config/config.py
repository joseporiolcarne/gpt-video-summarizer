import os


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEMPLATE_DIR = os.path.join(PROJECT_DIR, "templates")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

DATA_DIR = os.path.join(PROJECT_DIR, "data")

TEMPORARY_DIR = os.path.join(PROJECT_DIR, "tmp")

SUMMARIZE_DEFAULT_DETAIL = 0.1

SUMMARIZE_DEFAULT_LANGUAGE = "english"

SUMMARIZE_DEFAULT_TEMPLATE = "markdown"

SUMMARIZE_CAREFUL_WORDS = "DALL-E, ChatGPT, OpenAI"
