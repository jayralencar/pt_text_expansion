# Tokenizer 
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')

# PyTorch
model_pt = T5ForConditionalGeneration.from_pretrained("./ptt5-text-expansion/")

def expand(sentence):
  tokenized = tokenizer(sentence,return_tensors="pt")
  outputs = model_pt.generate(tokenized['input_ids'])
  out = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return out

from tinydb import TinyDB, Query

db = TinyDB('db.json')
user_table = db.table('user')
sentence_table = db.table("sentence")
User = Query()

from flask import Flask
from flask import request,render_template,send_from_directory
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import uuid

app = Flask(__name__,template_folder='reaact-board/www')
run_with_ngrok(app)
CORS(app)

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
  return send_from_directory('./reaact-board/www/', path)


@app.route('/')
def hello():
    return render_template("index.html")
  
@app.route("/user",methods=['POST'])
def user():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
      json = request.json
  else:
      return 'Content-Type not supported!'

  body = json

  if "email" in body:
    us = user_table.search(User.email == body["email"])
    if len(us) > 0:
      return us[0]

  myuuid = uuid.uuid4()
  body['user_id'] = str(myuuid) 

  user_table.insert(body)

  return body

@app.route("/sentence",methods=['POST'])
def sentence():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
      json = request.json
  else:
      return 'Content-Type not supported!'

  body = json

  sentence_table.insert(body)

  return body

@app.route("/model",methods=['POST'])
def inference():
  content_type = request.headers.get('Content-Type')
  if (content_type == 'application/json'):
      json = request.json
  else:
      return 'Content-Type not supported!'

  body = json
  return {"expanded_sentence":expand(body['sentence'])}
  # method = body["method"]
  # if "param" in body:
  #   param = body["param"]
  # sentence = body['sentence']
  # if "return_dict" in body:
  #   if body["return_dict"] == True:
  #     return {"show":predict_dict(sentence,method,param)}
  
  # predicted = prediction(sentence,method,param)
  # return {"show":predicted}
  
if __name__ == "__main__":
  app.run()