import os
import sys
from flask import Flask
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module import RouteChain

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route(RouteChain().fuck.you.bitch.__str__(), methods=['GET', 'POST'])
def fuck():
    return 'fuck you bitch'

if __name__ == '__main__':
    app.run()
