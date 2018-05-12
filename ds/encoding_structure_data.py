import pandas as pd
import pinyin
import ds.common 

class DiagnoseData(object):

    def __init__(self, columnname, datalist):
        self.columnname = columnname
        self.datalist = datalist


class EnumData(DiagnoseData):
    def __init__(self, columnname, datalist, enums):
        super(EnumData, self).__init__(columnname, datalist)
        self.enums = enums

    def encoding2onehot(self):
        m = {}
        py = pinyin.get(self.columnname, format='strip').strip('()').replace(')', '-').replace('(', '')
        for i in range(len(self.enums)):
            m[py + '_' + str(i)] = [0] * len(self.datalist)
        res = pd.DataFrame(m)
        for idx, val in enumerate(self.datalist):
            for enmidx, enm in enumerate(self.enums):
                if val == enm:
                    res.loc[idx, py + '_' + str(enmidx)] = 1
                    break
        return res


class BlockData(DiagnoseData):
    def __init__(self, columnname, datalist, blocks, unit):
        super(BlockData, self).__init__(columnname, datalist)
        self.blocks = blocks
        self.unit = unit 

    def encoding2onehot(self):
        m = {}
        py = pinyin.get(self.columnname, format='strip').strip('()').replace(')', '-').replace('(', '')
        for i in range(len(self.blocks)):
            m[py + '_' + str(i)] = [0] * len(self.datalist)
        res = pd.DataFrame(m)
        for idx, val in enumerate(self.datalist):
            for blcidx, blc in enumerate(self.blocks):
                if blc[0] <= val < blc[1]:
                    res.loc[idx, py + '_' + str(blcidx)] = 1
                    break
        return res



def encoding(df):
    pendingjobs = []
    for key in ds.common.DATA_PARTITION_TABLE:
        val = ds.common.DATA_PARTITION_TABLE[key]
        if isinstance(val, tuple):
            pendingjobs.append(BlockData(key, df[key], val[0], val[1]))
        else:
            pendingjobs.append(EnumData(key, df[key], val))

    resdf = pd.DataFrame()
    for job in pendingjobs:
        subdf = job.encoding2onehot()
        resdf = pd.concat([resdf, subdf], axis=1)

    return resdf 

