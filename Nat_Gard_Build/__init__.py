from flask import Flask
import os

app = Flask(__name__)
app.config.from_object(__name__) 
app.config.update(dict(
    UPLOAD_FOLDER = "Nat_Gard_Build/static/tmp/",
    DATA_FOLDER = "Nat_Gard_Build/models/",
    ALLOWED_EXTENSIONS = set(['jpg','jpeg'])
    ))

from Nat_Gard_Build import views
