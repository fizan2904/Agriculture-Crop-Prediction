import os
from app import app
from flask import Flask, session

app.config.from_object(__name__)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0', port=port)