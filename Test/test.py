from datasets import load_dataset
from transformers import LEDTokenizer,AutoTokenizer
from transformers import LEDForConditionalGeneration
# tokenizer = LEDTokenizer.from_pretrained("hyesunyun/update-summarization-bart-large-longformer")
# model = LEDForConditionalGeneration.from_pretrained("hyesunyun/update-summarization-bart-large-longformer")
# tokenizer = AutoTokenizer.from_pretrained("t5-small")
billsum =load_dataset("billsum",split="ca_test")
billsum = billsum.train_test_split(test_size =0.2)
print(billsum["train"][0]["title"])