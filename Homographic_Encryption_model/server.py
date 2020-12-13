import threading
import collections
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy 
import uuid
from  werkzeug.security import generate_password_hash, check_password_hash 
import jwt 
from datetime import datetime, timedelta 
from functools import wraps
import requests
import json
import pickle
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import time

model = None
client_address = None
app = Flask(__name__)
headers = {'Content-Type':"application/json"}
HE_Cl = Pyfhel()
sema = threading.Semaphore(10)
# sema1 = threading.Semaphore()
start = 0

#--------------------------------------------------------------------------------------------------------------------------------
# Authentication
app.config['SECRET_KEY'] = 'your secret key'
# database name 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# creates SQLALCHEMY object 
db = SQLAlchemy(app) 

# Database ORMs 
class User(db.Model): 
	id = db.Column(db.Integer, primary_key = True) 
	public_id = db.Column(db.String(50), unique = True) 
	name = db.Column(db.String(100)) 
	email = db.Column(db.String(70), unique = True) 
	password = db.Column(db.String(80))

def token_required(f): 
    @wraps(f) 
    def decorated(*args, **kwargs): 
        token = None
        # jwt is passed in the request header 
        if 'x-access-token' in request.headers: 
            token = request.headers['x-access-token'] 
        # return 401 if token is not passed 
        if not token: 
            return jsonify({'message' : 'Token is missing !!'}), 401
   
        try: 
            # decoding the payload to fetch the stored details 
            data = jwt.decode(token, app.config['SECRET_KEY']) 
            current_user = User.query.filter_by(public_id = data['public_id']).first() 
        except: 
            return jsonify({ 
                'message' : 'Token is invalid !!'
            }), 401
        # returns the current logged in users contex to the routes 
        return  f(current_user, *args, **kwargs) 
   
    return decorated  

# User Database Route 
# this route sends back list of users users 
@app.route('/user', methods =['GET'])
@token_required
def get_all_users(current_user): 
	# querying the database 
	# for all the entries in it 
	users = User.query.all() 
	# converting the query objects 
	# to list of jsons 
	output = [] 
	for user in users: 
		# appending the user data json 
		# to the response list 
		output.append({ 
			'public_id': user.public_id, 
			'name' : user.name, 
			'email' : user.email 
		})

	return jsonify({'users': output}) 

# route for loging user in 
@app.route('/login', methods =['POST'])
def login(): 
	# creates dictionary of form data
	auth = request.form 

	if not auth or not auth.get('email') or not auth.get('password'): 
		# returns 401 if any email or / and password is missing 
		return make_response( 
			'Could not verify', 
			401, 
			{'WWW-Authenticate' : 'Basic realm ="Login required !!"'} 
		) 

	user = User.query.filter_by(email = auth.get('email')).first() 

	if not user: 
		# returns 401 if user does not exist 
		return make_response( 
			'Could not verify', 
			401, 
			{'WWW-Authenticate' : 'Basic realm ="User does not exist !!"'} 
		)

	if check_password_hash(user.password, auth.get('password')): 
		# generates the JWT Token 
		token = jwt.encode({ 
			'public_id': user.public_id, 
			'exp' : datetime.utcnow() + timedelta(minutes = 30) 
		}, app.config['SECRET_KEY']) 

		return make_response(jsonify({'token' : token.decode('UTF-8')}), 201) 
	# returns 403 if password is wrong 
	return make_response( 
		'Could not verify', 
		403, 
		{'WWW-Authenticate' : 'Basic realm ="Wrong Password !!"'} 
	) 

# signup route 
@app.route('/signup', methods =['POST'])
def signup(): 
	# creates a dictionary of the form data 
	data = request.form 

	# gets name, email and password 
	name, email = data.get('name'), data.get('email') 
	password = data.get('password') 

	# checking for existing user 
	user = User.query.filter_by(email = email).first() 
	if not user: 
		# database ORM object 
		user = User( 
			public_id = str(uuid.uuid4()), 
			name = name, 
			email = email, 
			password = generate_password_hash(password) 
		) 
		# insert user 
		db.session.add(user) 
		db.session.commit() 

		return make_response('Successfully registered.', 201) 
	else: 
		# returns 202 if user already exists 
		return make_response('User already exists. Please Log in.', 202) 
#--------------------------------------------------------------------------------------------------------------------------------
# Authentication Completed

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def model_averaging(weight_list, bias_list,prev_weight_list,prev_bias_list):
        # temporary lists
    # org_weight_list = weight_list
    # org_bias_list = bias_list
    cipher = HE_Cl.encryptFrac(0)
    cipher1 = HE_Cl.encryptFrac(1)
    w_list = []
    b_list = []
        # updating weights and biases
    for i in range(len(weight_list)):
        W_a = weight_list[i]
        cipher.from_bytes(W_a,float)
        W_b = prev_weight_list[i]
        cipher1.from_bytes(W_b,float)
        W_c = (cipher + cipher1)/2
        W_c = W_c.to_bytes()
        w_list.append(W_c)

    for a in range(len(bias_list)):
        b_a = bias_list[a]
        cipher.from_bytes(b_a,float)
        b_b = prev_bias_list[a]
        cipher1.from_bytes(b_b,float)
        b_c = (cipher + cipher1)/2
        b_c = b_c.to_bytes()
        b_list.append(b_c)
    # print(compare(w_list,weight_list))
    return weight_list, bias_list

@app.route('/init', methods = ['POST'])
# @token_required
def sendtoclient():
    global prev_weight_list, prev_bias_list, start
    start = time.time()
    prev_bias_list = [] 
    prev_weight_list = []
    try_to_decrypt = 0
    #print(request.data)
    sema.acquire()
    received_data = pickle.loads(request.data)
    if "subject" in received_data:
        model =  received_data["model"]
        client_address = received_data["node_address"]
        print("Updated model sent to the client:", client_address)
        # received_data = pickle.loads(request.data)
        response = requests.post(client_address + "/train",
                                    data = request.data, headers = headers)
    sema.release()

    while(True):
        # sema1.acquire()

        if response.status_code == 200:
            response_data = pickle.loads(response.content)
            weight_list = response_data["weight_list"]
            bias_list = response_data["bias_list"]
            contx_file = response_data["contx_file"]
            pk_file = response_data["pk_file"]
            HE_Cl.restoreContext(contx_file)
            HE_Cl.restorepublicKey(pk_file)

            if try_to_decrypt == 0:

                # Attempting to decrypt results raises an error (missing secret key)
                #> ---------------------------------------------------------------------------
                #> RuntimeError                              Traceback (most recent call last)
                #> Pyfhel/Pyfhel.pyx in Pyfhel.Pyfhel.Pyfhel.decryptFrac()
                #> RuntimeError: Missing a Private Key [...]
                cipher2 = HE_Cl.encryptFrac(0)
                cipher2.from_bytes(weight_list[0],float)
                try:
                    print(HE_Cl.decrypt(cipher2))
                    raise Exception("This should not be reached!")
                except RuntimeError:
                    print("The cloud tried to decrypt, but couldn't as the private key is missing!")

            try_to_decrypt = 1
            print(len(prev_weight_list), len(prev_bias_list))
            #print(HE_Cl)
            if len(prev_weight_list) == 0:
                prev_weight_list = weight_list
                prev_bias_list = bias_list
                print("Updated model as prev list length is 0 and sent to the client:", client_address)
                response = requests.post(client_address + "/train",
                                            data = pickle.dumps(response_data), headers = headers)
            else:
                sema.acquire()
                updated_weight_list, updated_bias_list = model_averaging(weight_list, bias_list, prev_weight_list, prev_bias_list)
                prev_weight_list = updated_weight_list
                prev_bias_list = updated_bias_list
                #print(compare(prev_weight_list))
                print(len(updated_weight_list), len(updated_bias_list))
                data = {'weight_list':updated_weight_list,  'bias_list':updated_bias_list}
                #print('Updated parameters Sent to the client')
                data_byte = pickle.dumps(data)
                print("Updated parameters sent to the client:", client_address)
                response = requests.post(client_address + "/train",
                                            data = data_byte, headers = headers)
                end = time.time()
                print("time elapsed:", end - start)
                start = time.time()
                sema.release()
            # print(type(response_data))
    return "Registeration Successful", 200