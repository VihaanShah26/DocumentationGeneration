import pandas as pd

# Load JSONL file into a DataFrame
file_path = "trainData.json"
df = pd.read_json(file_path, lines=True)

# Display the DataFrame
rowNum = 0
print(df['rows'][0][rowNum]['row']['code_tokens']) 
print(df['rows'][0][rowNum]['row']['docstring_tokens'])  