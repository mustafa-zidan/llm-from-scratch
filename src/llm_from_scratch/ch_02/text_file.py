import urllib.request
import os


import urllib.request
import os

class TextFile:
    def __init__(self, url: str, file_path: str):
        self.url = url
        self.file_path = file_path
        self.raw_text = ""

    def download(self):
        """Download the file from the URL and save it locally."""
        urllib.request.urlretrieve(self.url, self.file_path)

    def load(self):
        """Load text from file into memory. Download if missing."""
        if not os.path.exists(self.file_path):
            self.download()
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        print(f"File: {self.file_path}")
        print(f"Length: {len(self.raw_text)} characters")
        print("\nFirst 500 characters:")
        print(self.raw_text[:500])
        print("\n... (truncated)")

    def char_count(self) -> int:
        if not self.raw_text:
            raise ValueError("No text loaded. Call load() first.")
        return len(self.raw_text)

    def preview(self, n: int = 100) -> str:
        if not self.raw_text:
            raise ValueError("No text loaded. Call load() first.")
        return self.raw_text[:n]


if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'the-verdict.txt')

    tf = TextFile(url, file_path)

    # you can either explicitly download, then load:
    tf.download()
    tf.load()

    # or just call load() and it will download if missing:
    tf.load()
    print("Total number of characters:", tf.char_count())
    print(tf.preview(99))