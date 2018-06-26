import json
import os
from pprint import pprint

this_dir = os.path.dirname(__file__)


with open(this_dir + "/relation_label.json", mode='rt') as f:
    relation_label = json.load(f)

with open(this_dir + "/entity_label.json", mode='rt') as f:
    entity_label = json.load(f)

def get_relation_id(relation):
    return str(relation_label[relation])

def get_entity_id(entity):
    return str(entity_label[entity])

def make_schema_file():
    entity_type = list(entity_label.keys())

    features = []
    
    ct = "col_type"
    cn = "col_name"
    
    features.append({ct: "string", cn: "entity1_name"})
    features.append({ct: "float",  cn: "entity1_start", "unit": None})
    features.append({ct: "float",  cn: "entity1_end",   "unit": None})
    features.append({ct: "enum",   cn: "entity1_type",  "options": entity_type})

    features.append({ct: "string", cn: "entity2_name"})
    features.append({ct: "float",  cn: "entity2_start", "unit": None})
    features.append({ct: "float",  cn: "entity2_end",   "unit": None})
    features.append({ct: "enum",   cn: "entity2_type",  "options": entity_type})

    features.append({ct: "string", cn: "sentence"})

    ret = {}

    ret["labels"] = list(relation_label.keys())
    ret["features"] = features


    with open("/models/relation_cnn/schema", mode='wt') as f:
        json.dump(ret, f, indent=2)

    return ret

if __name__ == "__main__":
    pprint(make_schema_file())
