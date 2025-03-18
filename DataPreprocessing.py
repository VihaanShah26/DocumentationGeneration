import pandas as pd 
import json

file_path = "testData.json"
df = pd.read_json(file_path, lines=True)

lines = [] 

file = open('ProcessedTestData.json', 'w')

for rowNum in range(len(df['rows'][0])):
    line = {'code_tokens': df['rows'][0][rowNum]['row']['code_tokens'], 
            'docstring_tokens': df['rows'][0][rowNum]['row']['docstring_tokens'],
            'code': df['rows'][0][rowNum]['row']['code'], 
            'docstring': df['rows'][0][rowNum]['row']['docstring']}
    json.dump(line, file)
    file.write('\n')

file.close()