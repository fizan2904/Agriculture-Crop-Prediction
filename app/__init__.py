from flask import Flask, session, render_template, redirect

app = Flask(__name__, static_url_path='')
app.debug = True

from app.routes import *

@app.route('/')
def index():
	return render_template('index.html')