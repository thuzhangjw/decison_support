from flask import Flask
import os 
import tensorflow as tf 
import ds.common  
import json
import ds.prediction 
from flask import request
from flask import jsonify 
import common 


app = Flask(__name__)

ds_model_path = os.path.join(common.MODEL_DIR, ds.common.DS_MODEL_NAME)
ds_sess, ds_model = ds.prediction.load_model(os.path.join(ds_model_path, 'checkpoints'), os.path.join(ds_model_path, 'schema'))


@app.route('/models/')
def get_models():
    print('\033[1;32mRequest: get models\033[0m')
    models = os.listdir(common.MODEL_DIR)
    return jsonify({'models': models})


@app.route('/<model_name>/schema')
def get_model_schema(model_name):
    print('\033[1;32mRequest: get model schema\033[0m')
    schema_path = os.path.join(common.MODEL_DIR, model_name, 'schema')
    schema = json.load(open(schema_path, 'r'))
    return jsonify(schema)


@app.route('/<model_name>/predict', methods=['POST'])
def predict(model_name):
    print('\033[1;32mRequest: predict \033[0m')
    if model_name == ds.common.DS_MODEL_NAME:
        data = json.loads(request.form['data'])
        probs = ds.prediction.predict(ds_sess, ds_model, data['现病史'], data)
        reslist = []
        for i in range(len(prob)):
            tmp = {}
            for j in range(len(probs[i])):
                tmp[ds.common.CLASSES[j]] = probs[i][j]
            reslist.append(tmp)
        return jsonify({'probabilities' : reslist})

