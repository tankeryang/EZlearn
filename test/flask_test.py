import os
import sys
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# customer module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module import RouteChain
from config import ALLOCATE_DTYPE_PARAMS, TRANSFORM_DTYPE_PARAMS
from config import SQL_SELECT_FROM_ALLOCATE_FETURE, SQL_SELECT_FROM_TRANSFORM_FETURE


app = Flask(__name__)
CORS(app=app)

train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train'))
model_allocate_file_name = 'mms_model_allocate.txt'
model_transform_file_name = 'mms_model_transform.txt'
model_allocate_file = os.path.join(train_path, model_allocate_file_name)
model_transform_file = os.path.join(train_path, model_transform_file_name)
model_allocate = lgb.Booster(model_file=model_allocate_file)
model_transform = lgb.Booster(model_file=model_transform_file)

presto_engine = create_engine("presto://prod@10.10.22.8:10300/prod_hive/ods_predict")


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route(RouteChain().signin.__str__(), methods=['GET'])
def signin_form():
    return render_template('form.html')


@app.route(RouteChain().signin.__str__(), methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    if username=='admin' and password=='password':
        return render_template('signin-ok.html', username=username)
    return render_template('form.html', message='Bad username or password', username=username)


@app.route(RouteChain().mms.allocate.__str__(), methods=['GET', 'POST'])
def predict_allocate():
    store_code = request.args.get('store')
    skc_code = request.args.get('skc')

    sql = SQL_SELECT_FROM_ALLOCATE_FETURE.format(store_code=store_code, skc_code=skc_code)
    
    con = presto_engine.connect()
    input_df = pd.read_sql_query(sql=sql, con=con)
    for col in ALLOCATE_DTYPE_PARAMS:
        input_df[col] = input_df[col].astype(ALLOCATE_DTYPE_PARAMS[col])

    result_data = model_allocate.predict(input_df)

    return jsonify(result=list(result_data))


@app.route(RouteChain().mms.transform.__str__(), methods=['GET', 'POST'])
def predict_transform():
    store_code = request.args.get('store')
    skc_code = request.args.get('skc')

    sql = SQL_SELECT_FROM_TRANSFORM_FETURE.format(store_code=store_code, skc_code=skc_code)
    
    con = presto_engine.connect()
    input_df = pd.read_sql_query(sql=sql, con=con)
    for col in TRANSFORM_DTYPE_PARAMS:
        input_df[col] = input_df[col].astype(TRANSFORM_DTYPE_PARAMS[col])

    result_data = model_transform.predict(input_df)

    return jsonify(result=list(result_data))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
