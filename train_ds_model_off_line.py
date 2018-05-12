import pandas as pd
import sys 
import json
import ds.encoding_structure_data as dse  
import ds.processing_unstructure_data as dsp 
import ds.common as dsc 
import ds.decision_support_model as dsd 
import os 


df = pd.read_excel(sys.argv[1], dtype=object)

print('\033[1;33mencode structure data\033[0m')
structure_df = dse.encoding(df)

print('\033[1;33mprocess unstructure data\033[0m')
dsp.text_keyword_mapping(df[dsc.CMH_NAME])
unstructure_df = dsp.text_participle(df[dsc.CMH_NAME])
word2vec_model = dsp.word2vec('./ds/word2vec.model')

print('\033[1;33mgenerate model schema\033[0m')
schema = dsc.generate_schema()

print('\033[1;33mtrain model\033[0m')
newdf = pd.concat([df[dsc.LABEL_NAME], unstructure_df, structure_df], axis=1)
model_name = dsc.DS_MODEL_NAME 
model_dir = dsc.MODEL_DIR
max_words_num, max_sentences_num, max_words_num_per_sentence = dsd.train(newdf, model_name, word2vec_model, model_dir, epochs=2)

schema['max_words_num'] = max_words_num
schema['max_sentences_num'] = max_sentences_num 
schema['max_words_num_per_sentence'] = max_words_num_per_sentence 

schema_path = os.path.join(model_dir, model_name, 'schema') 
json.dump(schema, open(schema_path, 'w'), ensure_ascii=False, indent=4)

