import json
from itertools import chain
import sys, os, pickle
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from cnn_model import cnn_model
from data_helper import test_data_generation, relation_label_dict, eval_relation_id, eval_entity_id, test_data_parse
from config_helper import test_config_parser, __MODEL_DIR__
from data.make_schema import get_relation_id, get_entity_id
from pprint import pprint

def test_result_parser(two_entity, sentence, relation):
    if relation == 'other':
        return None
    else:
        (e1, *_, e1_type, e1_index), (e2, *_, e2_type, e2_index) = two_entity
        if e1_type == e2_type:
            return None
        ret = { "relation_id": get_relation_id(relation), 
                "relation_name": relation,
                "from_class_id": get_entity_id(e1_type),
                "from_class_name": e1_type,
                "to_class_id": get_entity_id(e2_type),
                "to_class_name": e2_type,
                "relation_list":[{
                        "from_entity": e1,
                        "from_entity_index": str(e1_index),
                        "to_entity": e2,
                        "to_entity_index": str(e2_index)
                        }]
                }
        return ret


def test_routine(args):
    (seq_len, data_blocks, data_dicts,
     data_dict_sizes, word_vector, entity_pair_list, sent_list) = test_data_generation(args)
    print(seq_len)
    model = cnn_model(model_name="",
                      seq_len=seq_len,  # length of largest sent
                      reuse=args.reuse,
                      data_dict_sizes=data_dict_sizes,
                      word_vector=word_vector,
                      args=args)  # wordEmbedding
    predicted = model.predict(data_blocks)
    num_2_label = {v: k for k, v in relation_label_dict.items()}
    relations_predicted = [num_2_label[y] for y in predicted]
    results = []
    for i, r in enumerate(relations_predicted):
        relation_instance = test_result_parser(two_entity=entity_pair_list[i],
                                               sentence=sent_list[i],
                                               relation=r)
        if relation_instance is None:
            continue
        results.append(relation_instance)
    pprint(results)
    dp = {"result": results}
    with open(args.test_output_file, mode='w', encoding='GBK') as f:
        json.dump(dp, f, indent=2)


def get_test_model():
    """
    get the model based on the training information
    """
    args = test_config_parser.parse_args()
    model_file_path = os.path.join(__MODEL_DIR__, 'model', args.reuse)
    shape_file_path = model_file_path + ".pickle"
    with open(shape_file_path, mode='rb') as f:
        seq_len, data_dict_sizes, *_ = pickle.load(f)
    
    model = cnn_model(model_name="",
                      seq_len=9999,
                      reuse=args.reuse,
                      data_dict_sizes=data_dict_sizes,
                      word_vector=None, 
                      args=args)
    return model

def data_converter(data_input):
    class_name_mapper = {
        "symptom": "complaintsymptom",
        "examination": "testresult",
        "disease": "disease",
        "medication": "treatment"
    }
    sentence = data_input['originalText']
    ent_class_list = data_input['class']
    def generate_from_entity_class(ent_class):
        ent_pool = []
        ent_type = class_name_mapper[ent_class['class_name']]
        for ent_ele in ent_class['entity_list']:
            ent_name = ent_ele['entity']
            ent_start = int(ent_ele['entity_index'])
            ent_end = ent_start + len(ent_name)
            ent_pool.append({
                "entity_name": ent_name,
                "entity_start": ent_start,
                "entity_end": ent_end,
                "entity_type": ent_type
                })
        return ent_pool
    ent_list = list(chain.from_iterable(list(map(generate_from_entity_class, ent_class_list))))
    total_ents_num = len(ent_list)
    data_output = {}
    data_output['sentence'] = []
    data_output['entity1_name'] = []
    data_output['entity2_name'] = []
    data_output['entity1_start'] = []
    data_output['entity2_start'] = []
    data_output['entity1_end'] = []
    data_output['entity2_end'] = []
    data_output['entity1_type'] = []
    data_output['entity2_type'] = []
    for i in range(total_ents_num):
        ent1 = ent_list[i]
        for j in range(i+1, total_ents_num):
            ent2 = ent_list[j]
            data_output['sentence'].append(sentence)
            data_output['entity1_name'].append(ent1['entity_name'])
            data_output['entity2_name'].append(ent2['entity_name'])
            data_output['entity1_start'].append(ent1['entity_start'])
            data_output['entity2_start'].append(ent2['entity_start'])
            data_output['entity1_end'].append(ent1['entity_end'])
            data_output['entity2_end'].append(ent2['entity_end'])
            data_output['entity1_type'].append(ent1['entity_type'])
            data_output['entity2_type'].append(ent2['entity_type'])
    return data_output


def predict(model, data, myjieba):
    """
    make the prediction based on the model and the data
    """
    args = test_config_parser.parse_args()
    (seq_len, data_blocks, data_dicts, data_dict_sizes, 
     word_vector, entity_pair_list, sent_list) = test_data_parse(data_converter(data), myjieba, args)
    assert word_vector is None, "You are trying to USE the pretrained word embedding, not implemented"
    predicted = model.predict(data_blocks)
    num_2_label = {v: k for k, v in relation_label_dict.items()}
    relations_predicted = [num_2_label[y] for y in predicted]
    relation_instance_list = []
    for i, r in enumerate(relations_predicted):
        relation_instance = test_result_parser(two_entity=entity_pair_list[i],
                                               sentence=sent_list[i],
                                               relation=r)
        if relation_instance is not None:
            relation_instance_list.append(relation_instance)
    relation_count = len(relation_instance_list)
    relation_list = []
    def get_meta(relation_instance):
        meta_str = (relation_instance['relation_id'] +
                    relation_instance['relation_name'] +
                    relation_instance['from_class_id'] +
                    relation_instance['to_class_id'] +
                    relation_instance['to_class_name'])
        return meta_str
    if relation_instance_list:
        relation_dict_by_meta = {}
        for ri in relation_instance_list:
            meta = get_meta(ri)
            if meta not in relation_dict_by_meta.keys():
                relation_dict_by_meta[meta] = [ri]
            else:
                relation_dict_by_meta[meta].append(ri)
        relation_group_concat_list = []
        for _, relation_group in relation_dict_by_meta.items():
            relation_group_concat = {}
            first_re = relation_group[0]
            relation_group_concat['relation_id'] = first_re['relation_id']
            relation_group_concat['relation_name'] = first_re['relation_name']
            relation_group_concat['from_class_id'] = first_re['from_class_id']
            relation_group_concat['from_class_name'] = first_re['from_class_name']
            relation_group_concat['to_class_id'] = first_re['to_class_id']
            relation_group_concat['to_class_name'] = first_re['to_class_name']
            relation_group_concat['relation_list'] = []
            for r in relation_group:
                relation_group_concat['relation_list'].append(r['relation_list'])
            relation_group_concat_list.append(relation_group_concat)
        dp = {"statusCode": 200,
                "statusData": {
                    "relationType": '0001',
                    "relationCount": str(relation_count),
                    "relation": relation_group_concat_list
                    }}
        return dp
    else:
        return {"statusCode": 400, "statusData": None}


if __name__ == "__main__":
    test_routine(test_config_parser.parse_args())

