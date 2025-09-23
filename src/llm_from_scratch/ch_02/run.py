
import  os
from llm_from_scratch.ch_02.text_file import TextFile
from llm_from_scratch.ch_02.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'the-verdict.txt')
    tf = TextFile(url, file_path)
    tf.load()
    print(">>>>>>>>>>>>>>>>>>SimpleTokenizerV1>>>>>>>>>>>>>>>>>>>")
    # build vocab
    tokenizer = SimpleTokenizerV1(tf.raw_text)
    text = """"It's the last he painted, you know," 
     Mrs. Gisburn said with pardonable pride."""
    print("SimpleTokenizerV1: Encoding Text")
    ids = tokenizer.encode(text)
    print(ids)

    print("SimpleTokenizerV1: Decoding Ids to tokens")
    print(tokenizer.decode(ids))

    print("SimpleTokenizerV1: Encoding including Unknown Tokens ")
    try:
        text = "Hello, do you like tea?"
        ids = tokenizer.encode(text)
        print(ids)
    except Exception as e:
        print(f"Error encodin text: {e}")
    print(">>>>>>>>>>>>>>>>>>SimpleTokenizerV2>>>>>>>>>>>>>>>>>>>")
    tokenizer = SimpleTokenizerV2(tf.raw_text)
    text = """"It's the last he painted, you know," 
     Mrs. Gisburn said with pardonable pride."""

    print("SimpleTokenizerV2: Encoding Text")
    ids = tokenizer.encode(text)
    print(ids)

    print("SimpleTokenizerV2: Decoding Ids to tokens ")
    print(tokenizer.decode(ids))
    print("SimpleTokenizerV2: Encoding including Unknown Tokens ")
    text = "Hello, do you like tea?"
    ids = tokenizer.encode(text)
    print(ids )
    print("SimpleTokenizerV2: Decoding including Unknown Tokens ")
    print(tokenizer.decode(ids))

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
