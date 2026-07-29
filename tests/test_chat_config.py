import os
import unittest

from dotenv import load_dotenv


class ChatConfigTest(unittest.TestCase):
    def test_groq_api_key_is_loaded_from_dotenv(self):
        load_dotenv()
        self.assertTrue(os.getenv("GROQ_API_KEY"), "GROQ_API_KEY should be loaded from .env")


if __name__ == "__main__":
    unittest.main()
