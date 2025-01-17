import tiktoken


with open("the-verdict.txt", "r") as file:
    verdict = file.read()

print(len(verdict))  # 1000
print(verdict[:1000])

start_of_the_verdict = verdict[:1000]

tokenizer = tiktoken.get_encoding("gpt2")

integers = tokenizer.encode(start_of_the_verdict, allowed_special={"<|endoftext|>"})
print(integers)

text_from_int = tokenizer.decode(integers)

print(text_from_int)