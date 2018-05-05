import pandas as pd
import sys 
import json
import encoding_structure_data 


df = pd.read_excel(sys.argv[1], dtype=object)

print('\033[1;33mencode structure data\033[0m')
structure_df = encoding_structure_data.encoding(df)


print('\033[1;33mgenerate model schema\033[0m')
schema = generate_schema(pendingjobs, classes)


resdf = pd.concat([df['GB/T-codename'], df['现病史'], resdf], axis=1)
resdf.to_csv('../data/encodingdata.txt', sep='\t', index=False)

