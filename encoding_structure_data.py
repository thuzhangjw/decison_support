import pandas as pd
import pinyin
import common  

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


def generate_schema(diagnoseData_class_list, labels):
    schema = {'label': labels}
    features = [{'col_name': '现病史', 'col_type': 'string'}]
    for diagnose_class in diagnoseData_class_list:
        if isinstance(diagnose_class, EnumData):
            feature = {
                    'col_name': diagnose_class.columnname,
                    'col_type': 'enum',
                    'options': diagnose_class.enums 
                    }
            features.append(feature)
        else:
            feature = {
                    'col_name': diagnose_class.columnname,
                    'col_type': 'float',
                    'unit': diagnose_class.unit  
                    }
            features.append(feature)
    schema['features'] = features
    return schema 



def encoding(df):
    pendingjobs = [
        EnumData('性别', df['性别'], Data_Partition_Table['性别']),
        BlockData('年龄', df['年龄'], Data_Partition_Table['年龄'][0], Data_Partition_Table['年龄'][1]),
        EnumData('血型', df['血型'], Data_Partition_Table['血型']),
        BlockData('舒张压', df['舒张压'], Data_Partition_Table['舒张压'][0], Data_Partition_Table['舒张压'][1]), 
        BlockData('收缩压', df['收缩压'], Data_Partition_Table['收缩压'][0], Data_Partition_Table['收缩压'][1]),
        EnumData('心脏心律', df['心脏心律'], Data_Partition_Table['心脏心律']),
        EnumData('心音s1', df['心音s1'], Data_Partition_Table['心音s1']),
        EnumData('心音s2', df['心音s2'], Data_Partition_Table['心音s2']),
        EnumData('心音s3', df['心音s3'], Data_Partition_Table['心音s3']),
        EnumData('心音s4', df['心音s4'], Data_Partition_Table['心音s4']),
        EnumData('心音a2', df['心音a2'], Data_Partition_Table['心音a2']),
        EnumData('心音p2', df['心音p2'], Data_Partition_Table['心音p2']),
        EnumData('心音a2和p2关系', df['心音a2和p2关系'], Data_Partition_Table['心音a2和p2关系']),
        BlockData('心率', df['心率'], Data_Partition_Table['心率'][0], Data_Partition_Table['心率'][1]),
        BlockData('(WBC)白细胞', df['(WBC)白细胞_G/L'], Data_Partition_Table['(WBC)白细胞'][0], Data_Partition_Table['(WBC)白细胞'][1]),
        BlockData('(PLT)血小板', df['(PLT)血小板_G/L'], Data_Partition_Table['(PLT)血小板'][0], Data_Partition_Table['(PLT)血小板'][1]),
        BlockData('(RBC)红细胞计数', df['(RBC)红细胞计数_T/L'], Data_Partition_Table['(RBC)红细胞计数'][0], Data_Partition_Table['(RBC)红细胞计数'][1]),
        BlockData('(MCHC)平均红细胞Hb浓度', df['(MCHC)平均红细胞Hb浓度_g/L'], Data_Partition_Table['(MCHC)平均红细胞Hb浓度'][0], Data_Partition_Table['(MCHC)平均红细胞Hb浓度'][1]),
        BlockData('(HB)血红蛋白', df['(HB)血红蛋白_g/dl'], Data_Partition_Table['(HB)血红蛋白'][0], Data_Partition_Table['(HB)血红蛋白'][1]),
        BlockData('(LYMBFB)淋巴细胞百分比', df['(LYMBFB)淋巴细胞百分比_%'], Data_Partition_Table['(LYMBFB)淋巴细胞百分比'][0], Data_Partition_Table['(LYMBFB)淋巴细胞百分比'][1]),
        BlockData('(MONBFB)单核细胞百分比', df['(MONBFB)单核细胞百分比_%'], Data_Partition_Table['(MONBFB)单核细胞百分比'][0], Data_Partition_Table['(MONBFB)单核细胞百分比'][1]),
        BlockData('(NE)中性粒细胞数', df['(NE)中性粒细胞数_G/L'], Data_Partition_Table['(NE)中性粒细胞数'][0], Data_Partition_Table['(NE)中性粒细胞数'][1]),
        BlockData('(MON)单核细胞数', df['(MON)单核细胞数_G/L'], Data_Partition_Table['(MON)单核细胞数'][0], Data_Partition_Table['(MON)单核细胞数'][1]),
        BlockData('(HCT)红细胞压积', df['(HCT)红细胞压积_%'], Data_Partition_Table['(HCT)红细胞压积'][0], Data_Partition_Table['(HCT)红细胞压积'][1]),
        BlockData('(MCV)红细胞平均体积', df['(MCV)红细胞平均体积_fL'], Data_Partition_Table['(MCV)红细胞平均体积'][0], Data_Partition_Table['(MCV)红细胞平均体积'][1]),
        BlockData('(RDWSD)红细胞分布宽度-SD值', df['(RDWSD)红细胞分布宽度-SD值_fl'], Data_Partition_Table['(RDWSD)红细胞分布宽度-SD值'][0], Data_Partition_Table['(RDWSD)红细胞分布宽度-SD值'][1]),
        BlockData('(RDWCV)红细胞分布宽度-CV值', df['(RDWCV)红细胞分布宽度-CV值_%'], Data_Partition_Table['(RDWCV)红细胞分布宽度-CV值'][0], Data_Partition_Table['(RDWCV)红细胞分布宽度-CV值'][1]),
        BlockData('(MPV)平均血小板体积', df['(MPV)平均血小板体积_fL'], Data_Partition_Table['(MPV)平均血小板体积'][0], Data_Partition_Table['(MPV)平均血小板体积'][1]),
        BlockData('(EOSBFB)嗜酸细胞百分比', df['(EOSBFB)嗜酸细胞百分比_%'], Data_Partition_Table['(EOSBFB)嗜酸细胞百分比'][0], Data_Partition_Table['(EOSBFB)嗜酸细胞百分比'][1]),
        BlockData('(BAS)嗜碱细胞数', df['(BAS)嗜碱细胞数_G/L'], Data_Partition_Table['(BAS)嗜碱细胞数'][0], Data_Partition_Table['(BAS)嗜碱细胞数'][1]),
        BlockData('(PLCR)大血小板比率', df['(PLCR)大血小板比率_%'], Data_Partition_Table['(PLCR)大血小板比率'][0], Data_Partition_Table['(PLCR)大血小板比率'][1]),
        BlockData('(BUN)尿素', df['(BUN)尿素_mmol/L'], Data_Partition_Table['(BUN)尿素'][0], Data_Partition_Table['(BUN)尿素'][1]),
        BlockData('(CREA)肌酐', df['(CREA)肌酐_mmol/L'], Data_Partition_Table['(CREA)肌酐'][0], Data_Partition_Table['(CREA)肌酐'][1]),
        BlockData('(HDL)高密度脂蛋白', df['(HDL)高密度脂蛋白_mmol/L'], Data_Partition_Table['(HDL)高密度脂蛋白'][0], Data_Partition_Table['(HDL)高密度脂蛋白'][1]),  # 0.7--2.0
        BlockData('(LDL)低密度脂蛋白', df['(LDL)低密度脂蛋白_mmol/L'], Data_Partition_Table['(LDL)低密度脂蛋白'][0], Data_Partition_Table['(LDL)低密度脂蛋白'][1]),  # 0--3.12
        BlockData('(TC)总胆固醇', df['(TC)总胆固醇_mmol/L'], Data_Partition_Table['(TC)总胆固醇'][0], Data_Partition_Table['(TC)总胆固醇'][1]),  # 2.85--5.69
        BlockData('(TG)甘油三酯', df['(TG)甘油三酯_mmol/L'], Data_Partition_Table['(TG)甘油三酯'][0], Data_Partition_Table['(TG)甘油三酯'][1]),  #  0.45--1.69
        BlockData('(UA)尿酸', df['(UA)尿酸_umol/L'], Data_Partition_Table['(UA)尿酸'][0], Data_Partition_Table['(UA)尿酸'][1]),  # 89--416
        BlockData('钾(K)', df['钾(K)_mmol/L'], Data_Partition_Table['钾(K)'][0], Data_Partition_Table['钾(K)'][1]),  # 3.5--5.3
        BlockData('钠(NA)', df['钠(NA)_mmol/L'], Data_Partition_Table['钠(NA)'][0], Data_Partition_Table['钠(NA)'][1]),  # 135--145
        BlockData('氯(CL)', df['氯(CL)_mmol/L'], Data_Partition_Table['氯(CL)'][0], Data_Partition_Table['氯(CL)'][1]),  # 96--106
        BlockData('钙(CA)', df['钙(CA)_mmol/L'], Data_Partition_Table['钙(CA)'][0], Data_Partition_Table['钙(CA)'][1]),  # 2--2.54
        BlockData('镁(MG)', df['镁(MG)_mmol/L'], Data_Partition_Table['镁(MG)'][0], Data_Partition_Table['镁(MG)'][1]),  # 0.8--1
        BlockData('无机磷测定(P)', df['无机磷测定(P)_mmol/L'], Data_Partition_Table['无机磷测定(P)'][0], Data_Partition_Table['无机磷测定(P)'][1]),  # 0.7--1.45
        BlockData('(LDH)乳酸脱氢酶', df['(LDH)乳酸脱氢酶_U/L'], Data_Partition_Table['(LDH)乳酸脱氢酶'][0], Data_Partition_Table['(LDH)乳酸脱氢酶'][1]),  # 109--245
        EnumData('便潜血(BOBB)', df['便潜血(BOBB)'], Data_Partition_Table['便潜血(BOBB)']),
        BlockData('(AST)谷草转氨酶', df['(AST)谷草转氨酶_U/L'], Data_Partition_Table['(AST)谷草转氨酶'][0], Data_Partition_Table['(AST)谷草转氨酶'][1]),  # 0--40
        BlockData('(ALP)碱性磷酸酶', df['(ALP)碱性磷酸酶_U/L'], Data_Partition_Table['(ALP)碱性磷酸酶'][0], Data_Partition_Table['(ALP)碱性磷酸酶'][1]),  # 45--135
        BlockData('(TBA)血清总胆汁酸', df['(TBA)血清总胆汁酸_μmol/L'], Data_Partition_Table['(TBA)血清总胆汁酸'][0], Data_Partition_Table['(TBA)血清总胆汁酸'][1]),  # 0--12
        BlockData('(TP)总蛋白', df['(TP)总蛋白_g/L'], Data_Partition_Table['(TP)总蛋白'][0], Data_Partition_Table['(TP)总蛋白'][1]),  # 60--85
        BlockData('(ALB)白蛋白', df['(ALB)白蛋白_g/L'], Data_Partition_Table['(ALB)白蛋白'][0], Data_Partition_Table['(ALB)白蛋白'][1]),  # 35--55
        BlockData('(GA)糖化血清白蛋白', df['(GA)糖化血清白蛋白_%'], Data_Partition_Table['(GA)糖化血清白蛋白'][0], Data_Partition_Table['(GA)糖化血清白蛋白'][1]),  # 11--16
        BlockData('(DDIMER)D-二聚体定量', df['(DDIMER)D-二聚体定量_μg/l'], Data_Partition_Table['(DDIMER)D-二聚体定量'][0], Data_Partition_Table['(DDIMER)D-二聚体定量'][1]), # 0--l
        BlockData('(GLU)葡萄糖', df['(GLU)葡萄糖_mmol/L'], Data_Partition_Table['(GLU)葡萄糖'][0], Data_Partition_Table['(GLU)葡萄糖'][1]),  # 3.9--6.1
        BlockData('(PCO2)二氧化碳分压', df['(PCO2)二氧化碳分压_mmHg'], Data_Partition_Table['(PCO2)二氧化碳分压'][0], Data_Partition_Table['(PCO2)二氧化碳分压'][1]),  # 35-45
        BlockData('(SO2)氧饱和度', df['(SO2)氧饱和度_%'], Data_Partition_Table['(SO2)氧饱和度'][0], Data_Partition_Table['(SO2)氧饱和度'][1]),  # 91.9--99
        BlockData('(BASBFB)嗜碱细胞百分比', df['(BASBFB)嗜碱细胞百分比_%'], Data_Partition_Table['(BASBFB)嗜碱细胞百分比'][0], Data_Partition_Table['(BASBFB)嗜碱细胞百分比'][1]),  # 0--1
        BlockData('(FBG)纤维蛋白原定量', df['(FBG)纤维蛋白原定量_g/l'], Data_Partition_Table['(FBG)纤维蛋白原定量'][0], Data_Partition_Table['(FBG)纤维蛋白原定量'][1]),  # 2--4
        BlockData('(HCY)同型半胱氨酸', df['(HCY)同型半胱氨酸_umol/l'], Data_Partition_Table['(HCY)同型半胱氨酸'][0], Data_Partition_Table['(HCY)同型半胱氨酸'][1]),  # 4-15.4
        BlockData('(NEBFB)中性粒细胞百分比', df['(NEBFB)中性粒细胞百分比_%'], Data_Partition_Table['(NEBFB)中性粒细胞百分比'][0], Data_Partition_Table['(NEBFB)中性粒细胞百分比'][1]),  # 50--70
    ]

    resdf = pd.DataFrame()
    for job in pendingjobs:
        subdf = job.encoding2onehot()
        resdf = pd.concat([resdf, subdf], axis=1)

    print('generate model schema')
    schema = generate_schema(pendingjobs, common.CLASSES)
