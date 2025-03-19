from datasets import load_dataset, DatasetDict
import json

dataset = load_dataset("Nan-Do/code-search-net-python", split='train')  

train_testvalid = dataset.train_test_split(test_size=0.3, seed=42)

test_valid = train_testvalid['test'].train_test_split(test_size=1/3, seed=42)

split_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

# Process each split
for split in split_dataset:
    with open(f'Processed{split.capitalize()}Data.json', 'w') as file:
        for example in split_dataset[split]:

            line = {
                'code_tokens': example['code_tokens'],
                'docstring_tokens': example['docstring_tokens'],
                'code' : example['code'],
                'docstring' : example['docstring']
            }

            json.dump(line, file)
            file.write('\n')