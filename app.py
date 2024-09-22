from flask import Flask;
from flask_cors import CORS

app = Flask(__name__);

CORS(app)


app.secret_key = "1234";
app.app_context().push()

from controllers import *

if __name__ == "__main__":
    app.debug = True;
    app.run()