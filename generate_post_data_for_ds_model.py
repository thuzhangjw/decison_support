import pandas as pd
import sys
import ds.common
import json

df = pd.read_excel(sys.argv[1], dtype=object)
rows_num = int(sys.argv[2])

post_dict = {}

rows = df.iloc[0:rows_num, :]

for key in ds.common.DATA_PARTITION_TABLE:
    post_dict[key] = list(rows[key])

post_dict[ds.common.CMH_NAME] = list(rows[ds.common.CMH_NAME])
labels = list(rows[ds.common.LABEL_NAME]) 

with open('ds_model_post_data.txt', 'w') as f:
    f.write(json.dumps({'data': post_dict}, ensure_ascii=False, indent=4))
    f.write('\n\n\n')
    for label in labels:
        f.write(label+'\n')

