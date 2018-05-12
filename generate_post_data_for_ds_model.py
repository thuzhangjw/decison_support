import pandas as pd
import sys
import ds.common

df = pd.read_excel(sys.argv[1], dtype=object)
rows_num = int(sys.argv[1])

post_dict = {}

rows = df.iloc[0:rows_num, :]

for key in ds.common.DATA_PARTITION_TABLE:
    val = rows[key]

