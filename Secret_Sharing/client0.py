import numpy
from flask import Flask, request
import requests, json, pandas, backprop as bp, pickle
import requests
import json
import os
import random
import time
import threading

#---------------------------------------------------------------------------------------------------------------------------------------------
# Data Distribution

sema = threading.Semaphore()
df = pandas.read_csv('data.csv')
data_len = os.environ.get("data")
if data_len == 223:
    data = df[:int(len(df)/6)]
else:
    data = df[int(3*len(df)/6):int(4*len(df)/6)]
X = data.drop('charges', axis=1)
y = data['charges']
y = numpy.array(y)
y = y.reshape((len(y), 1))
data_inputs = numpy.array(X)
num_classes = 1
num_inputs = 12
thr = 1.5
data_outputs = numpy.array(y)
data_inputs = data_inputs.T
data_outputs = data_outputs.T
mean = numpy.mean(data_inputs, axis=1, keepdims=True)
std_dev = numpy.std(data_inputs, axis=1, keepdims=True)
data_inputs = (data_inputs - mean) / std_dev

description = [{"num_nodes" : 12, "activation" : "relu"},
               {"num_nodes" : 1, "activation" : "relu"}]

model = bp.NeuralNetwork(description, num_inputs, 'mean_squared', data_inputs, data_outputs, learning_rate=0.001)
s_model = bp.NeuralNetwork(description, num_inputs, 'mean_squared', data_inputs, data_outputs, learning_rate=0.001)
encrypted_weight_list = []
encrypted_bias_list = []
weight_dat = []
bias_dat = []
model.data = data_inputs
model.labels = data_outputs
history = model.train(1000)
prediction = model.layers[(-1)].a
honest = os.environ.get("honest")
error = model.calc_accuracy(data_inputs, data_outputs, 'RMSE')
print("RMSE", error)

#---------------------------------------------------------------------------------------------------------------------------------------------
# Data distribution

# Address Connection
count = 0
CONNECTED_NODE_ADDRESS = os.environ.get('peers')
CONNECTED_NODE_ADDRESS = list(CONNECTED_NODE_ADDRESS.split(" "))
myaddress = os.environ['myaddress']
# print(myaddress)
# print("Done bhai", CONNECTED_NODE_ADDRESS)
ready_to_send = set()
weight_data = []
bias_data = []
weight_list = []
bias_list = []
# data1 = [11,12,13]
# print(data1)


app = Flask(__name__)


#---------------------------------------------------------------------------------------------------------------------------------------------
#Shamir Started

from copy import copy
import random
PRIME = 19997
def base_egcd(a, b):
    r0, r1 = a, b
    s0, s1 = 1, 0
    t0, t1 = 0, 1
    
    while r1 != 0:
        q, r2 = divmod(r0, r1)
        r0, s0, t0, r1, s1, t1 = \
            r1, s1, t1, \
            r2, s0 - s1*q, t0 - t1*q

    d = r0
    s = s0
    t = t0
    return d, s, t
def base_inverse(a):
    _, b, _ = base_egcd(a, PRIME)
    return b if b >= 0 else b+PRIME
def base_add(a, b):
    return (a + b) % PRIME

def base_sub(a, b):
    return (a - b) % PRIME

def base_mul(a, b):
    return (a * b) % PRIME

def base_div(a, b):
    return base_mul(a, base_inverse(b))

def expand_to_match(A, B):
    diff = len(A) - len(B)
    if diff > 0:
        return A, B + [0] * diff
    elif diff < 0:
        diff = abs(diff)
        return A + [0] * diff, B
    else:
        return A, B

assert( expand_to_match([1,1], [])  == ([1,1], [0,0]) )
assert( expand_to_match([1,1], [1]) == ([1,1], [1,0]) )

def canonical(A):
    for i in reversed(range(len(A))):
        if A[i] != 0:
            return A[:i+1]
    return []

assert( canonical([ ]) == [] )
assert( canonical([0]) == [] )
assert( canonical([0,0]) == [] )
assert( canonical([0,1,2]) == [0,1,2] )
assert( canonical([0,1,2,0,0]) == [0,1,2] )

def lc(A):
    B = canonical(A)
    return B[-1]

assert( lc([0,1,2,0]) == 2 )

def deg(A):
    return len(canonical(A)) - 1

assert( deg([ ]) == -1 )
assert( deg([0]) == -1 )
assert( deg([1,0]) == 0 )
assert( deg([0,0,1]) == 2 )

def poly_add(A, B):
    F, G = expand_to_match(A, B)
    return canonical([ base_add(f, g) for f, g in zip(F, G) ])

assert( poly_add([1,2,3], [2,1]) == [3,3,3] )

def poly_sub(A, B):
    F, G = expand_to_match(A, B)
    return canonical([ base_sub(f, g) for f, g in zip(F, G) ])

assert( poly_sub([1,2,3], [1,2]) == [0,0,3] )

def poly_scalarmul(A, b):
    return canonical([ base_mul(a, b) for a in A ])

def poly_scalardiv(A, b):
    return canonical([ base_div(a, b) for a in A ])

def poly_mul(A, B):
    C = [0] * (len(A) + len(B) - 1)
    for i in range(len(A)):
        for j in range(len(B)):
            C[i+j] = base_add(C[i+j], base_mul(A[i], B[j]))
    return canonical(C)

def poly_divmod(A, B):
    t = base_inverse(lc(B))
    Q = [0] * len(A)
    R = copy(A)
    for i in reversed(range(0, len(A) - len(B) + 1)):
        Q[i] = base_mul(t, R[i + len(B) - 1])
        for j in range(len(B)):
            R[i+j] = base_sub(R[i+j], base_mul(Q[i], B[j]))
    return canonical(Q), canonical(R)

A = [7,4,5,4]
B = [1,0,1]
Q, R = poly_divmod(A, B)
#assert( poly_add(poly_mul(Q, B), R) == A )

def poly_div(A, B):
    Q, _ = poly_divmod(A, B)
    return Q

def poly_mod(A, B):
    _, R = poly_divmod(A, B)
    return R

def poly_eval(A, x):
    result = 0
    for coef in reversed(A):
        result = base_add(coef, base_mul(x, result))
    return result

def lagrange_polynomials(xs):
    polys = []
    for i, xi in enumerate(xs):
        numerator = [1]
        denominator = 1
        for j, xj in enumerate(xs):
            if i == j: continue
            numerator   = poly_mul(numerator, [base_sub(0, xj), 1])
            denominator = base_mul(denominator, base_sub(xi, xj))
        poly = poly_scalardiv(numerator, denominator)
        polys.append(poly)
    return polys

def lagrange_interpolation(xs, ys):
    ls = lagrange_polynomials(xs)
    poly = []
    for i in range(len(ys)):
        term = poly_scalarmul(ls[i], ys[i])
        poly = poly_add(poly, term)
    return poly

F = [1,2,3]

xs = [10,20,30,40]
ys = [ poly_eval(F, x) for x in xs ]

G = lagrange_interpolation(xs, ys)
#assert( G == F )


def poly_gcd(A, B):
    R0, R1 = A, B
    while R1 != []:
        R2 = poly_mod(R0, R1)
        R0, R1 = R1, R2
    D = poly_scalardiv(R0, lc(R0))
    return D

def poly_egcd(A, B):
    R0, R1 = A, B
    S0, S1 = [1], []
    T0, T1 = [], [1]
    
    while R1 != []:
        Q, R2 = poly_divmod(R0, R1)
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
            
    c = lc(R0)
    D = poly_scalardiv(R0, c)
    S = poly_scalardiv(S0, c)
    T = poly_scalardiv(T0, c)
    return D, S, T

A = [2,0,2]
B = [1,3]
G = [1,0,0,1]
assert( poly_gcd(A, B) == [1] )
assert( poly_gcd(A, G) == [1] )
assert( poly_gcd(B, G) == [1] )

F = poly_mul(G, A)
H = poly_mul(G, B)
D, S, T = poly_egcd(F, H)
assert( D == poly_gcd(F, H) )
assert( D == poly_add(poly_mul(F, S), poly_mul(H, T)) )

def poly_eea(F, H):
    R0, R1 = F, H
    S0, S1 = [1], []
    T0, T1 = [], [1]
    
    triples = []
    while R1 != []:
        Q, R2 = poly_divmod(R0, R1)
        
        triples.append( (R0, S0, T0) )
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
            
    return triples

def gao_decoding(points, values, max_degree, max_error_count):
    assert(len(values) == len(points))
    assert(len(points) >= 2*max_error_count + max_degree)
    
    # interpolate faulty polynomial
    H = lagrange_interpolation(points, values)
    
    # compute f
    F = [1]
    for xi in points:
        Fi = [base_sub(0, xi), 1]
        F = poly_mul(F, Fi)
    
    # run EEA-like algorithm on (F,H) to find EEA triple
    R0, R1 = F, H
    S0, S1 = [1], []
    T0, T1 = [], [1]
    while True:
        Q, R2 = poly_divmod(R0, R1)
        
        if deg(R0) < max_degree + max_error_count:
            G, leftover = poly_divmod(R0, T0)
            if leftover == []:
                decoded_polynomial = G
                error_locator = T0
                return decoded_polynomial, error_locator
            else:
                return None
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))

K = 1 # fixed in Shamir's scheme

N = 15
T = 5
R = T+K
assert(R <= N)

MAX_MISSING = 3
MAX_MANIPULATED = 2
assert(R + MAX_MISSING + 2*MAX_MANIPULATED <= N)

POINTS = [ p for p in range(1, N+1) ]
assert(0 not in POINTS)
assert(len(POINTS) == N)


def shamir_share(secret):
    polynomial = [secret] + [random.randrange(PRIME) for _ in range(T)]
    shares = [ poly_eval(polynomial, p) for p in POINTS ]
    return shares

def shamir_robust_reconstruct(shares):
    #assert(len(shares) == N)
    
    # filter missing shares
    points_values = [ (p,v) for p,v in zip(POINTS, shares) if v is not None ]
    assert(len(points_values) >= N - MAX_MISSING)
    
    # decode remaining faulty
    points, values = zip(*points_values)
    polynomial, error_locator = gao_decoding(points, values, R, MAX_MANIPULATED)
    
    # check if recovery was possible
    if polynomial is None: raise Exception("Too many errors, cannot reconstruct")

    # recover secret
    secret = poly_eval(polynomial, 0)
    
    # find error indices
    error_indices = [ i for i,v in enumerate( poly_eval(error_locator, p) for p in POINTS ) if v == 0 ]
    return secret, error_indices

#---------------------------------------------------------------------------------------------------------------------------------------------
#Shamir Completed

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing Shares List

def example_serialize(layers, weight_list, bias_list):
    encrypted_weight_list = []
    encrypted_bias_list = []
    m = 0
    n = 0
    # print(layers, len(weight_list[0][0]), len(bias_list[0]))
    for i in range(layers):
        for j in range(len(weight_list[i])):
            for k in range(len(weight_list[i][j])):
                encrypted_weight = weight_list[i][j][k]
                encrypted_weight = 1000*encrypted_weight
                encrypted_weight1 = shamir_share(int(encrypted_weight))
                for l in range(N):
                    encrypted_weight_list.append(encrypted_weight1[l])
                m = m + 1

        for a in range(len(bias_list[i])):
            for b in range(len(bias_list[i][a])):
                encrypted_bias = bias_list[i][a][b]
                encrypted_bias = 1000*encrypted_bias
                encrypted_bias1 = shamir_share(int(encrypted_bias))
                for z in range(N):
                    encrypted_bias_list.append(encrypted_bias1[z])
                n = n + 1
    #print(encrypted_weight_list[15:30])
    return encrypted_weight_list, encrypted_bias_list


def example_deserialize(weight_list1, bias_list1):
    decrypted_weight_list1 = []
    decrypted_bias_list1 = []
    decrypted_list = [[]]
    i = 0
    # cipher = HE.encryptFrac(i)
    #print("De-serializing",len(weight_list))
    while i < len(weight_list1) - num_inputs:
        j = 0
        temp_weight = []
        while j < num_inputs:
            temp_weight.append(weight_list1[i+j])
            j = j + 1
        decrypted_list.append(temp_weight)
        i = i + j
    decrypted_weight_list1.append(numpy.array(decrypted_list[1:]))
    j = 144
    temp_weight = []
    decrypted_list = [[]]
    while j < len(weight_list1):
            weight = weight_list1[j]
            temp_weight.append(weight)
            j = j + 1
    decrypted_list.append(temp_weight)
    decrypted_weight_list1.append(numpy.array(decrypted_list[1:]))
    # print("decrypted_weight_list1",decrypted_weight_list1)
    i = 0
    decrypted_list = [[]]
    while i < num_inputs:
        temp_bias = []
        bias = bias_list1[i]
        temp_bias.append(bias)
        decrypted_list.append(temp_bias)
        i = i + 1
    decrypted_bias_list1.append(numpy.array(decrypted_list[1:]))
    # print("decrypted_bias_list1",decrypted_bias_list1)
    temp_bias = []
    decrypted_list = [[]]
    bias = bias_list1[i]
    temp_bias.append(bias)
    decrypted_list.append(temp_bias)
    decrypted_bias_list1.append(numpy.array(decrypted_list[1:]))
    print("Reconstruction Done")
    # print("In func:", decrypted_weight_list1[0][0])
    return decrypted_weight_list1, decrypted_bias_list1

def Assign(weight_list, bias_list):
    NN_model = model
    for i in range(len(model.layers)):
        NN_model.layers[i].W = weight_list[i]
        NN_model.layers[i].b = bias_list[i]
    # print("weight of NN in Assign",NN_model.layers[0].W[0][0])
    return NN_model

def training(NN_model):
    NN_model.data = data_inputs
    NN_model.labels = data_outputs
    history = NN_model.train(1000)
    prediction = NN_model.layers[(-1)].a
    error = NN_model.calc_accuracy(data_inputs, data_outputs, 'RMSE')
    return (NN_model, error)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Completed  Shares List

# @app.route('/', methods = ['GET'])
# def init():
#     model = "nothing"
#     print("client calling server")
#     data = {"node_address": "http://127.0.0.1:5000","model":model}
#     headers = {'Content-Type':"application/json"}
#     response = requests.post("http://127.0.0.1:8000" + "/init",
#                                     data = json.dumps(data), headers = headers)
#     return json.dumps({"status":"Process Started"})


# @app.route('/train', methods = ['POST'])
# def train():
#     print("train is called by server")
#     incoming_model = request.get_json()["model"]
#     # train on model
#     sending_model = incoming_model
#     return json.dumps({"model":sending_model})

# @app.route('/add_peer', methods = ['POST'])
# def add_peer():
#     global CONNECTED_NODE_ADDRESS
#     print("Adding PEER")
#     peers = request.get_json()["peers"]
#     print(list(peers))
#     for peer in list(peers):
#         CONNECTED_NODE_ADDRESS.add(peer)
#     return "PEERS added Successfully", 200

@app.route('/get_peers', methods = ['GET'])
def get_peers():
    global CONNECTED_NODE_ADDRESS, data
    return json.dumps({"peers": list(CONNECTED_NODE_ADDRESS), "data": list(data)})

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Distributing Shares among clients
@app.route('/send_data', methods = ['GET'])
def send_data():
    while True:
        global data, CONNECTED_NODE_ADDRESS,encrypted_weight_list,encrypted_bias_list,weight_list,bias_list
         # sending
        # print("Initial 0th element",weight_list[0][0][0])
        # print("weight:", weight_list[0][0][0:12])
        # print("encrypted_weight_list", encrypted_weight_list)

        layers = len(model.layers)
        if len(encrypted_weight_list) == 0:
            for i in range(len(model.layers)):
                weight_list.append(model.layers[i].W)
                bias_list.append(model.layers[i].b)
            encrypted_weight_list, encrypted_bias_list = example_serialize(layers, weight_list, bias_list)
        # print("0th weight",weight_list[0])
        # print("Initial bias element",bias_list[0][0][0])
        for peer in CONNECTED_NODE_ADDRESS:
            if int(peer[-1]) == int(myaddress[-1]) + 1:
                l = 5
            else:
                l = 10
            encrypted_weight_shares = []
            encrypted_bias_shares = []
            for i in range(156):
                for k in range(l,l+5):
                    encrypted_weight_shares.append(encrypted_weight_list[k])
                l = l + N
            if int(peer[-1]) == int(myaddress[-1]) + 1:
                l = 5
            else:
                l = 10
            for i in range(13):
                for k in range(l,l+5):
                    encrypted_bias_shares.append(encrypted_bias_list[k])
                l = l + N
            # print("lenght of shares list:",len(encrypted_weight_shares))
            # print("lenght of shares list:",len(encrypted_bias_shares))
            url = "{}/recv_data".format(peer)
            sending_data = {'node_address': myaddress,  'encrypted_weight_shares': encrypted_weight_shares, 'encrypted_bias_shares': encrypted_bias_shares}
            # headers = {'Content-Type': 'application/json'}
            data_byte = pickle.dumps(sending_data)
            requests.post(url,
                        data= data_byte
                        )
        print('Distributed Weights to clients')
        #headers = {'x-access-token': token }
        # if response.status_code == 200:
        #     response_data = pickle.loads(response.content)
        #     recv_weight_shares = response_data["encrypted_weight_shares"]
        #     recv_bias_shares = response_data["encrypted_bias_shares"]
        #     return json.dumps({"Distribution Done"})
        # else:
        #     print(recv_bias_shares, recv_weight_shares)
        #     return data_byte
        time.sleep(20)
    return 'Sent', 200
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Distributed Shares among clients

@app.route('/recv_data', methods = ['POST'])
def recv_data():
    global data, ready_to_send , thr, weight_data, weight_dat, bias_dat, bias_data, encrypted_weight_list, encrypted_bias_list
    # for i in range(len(model.layers)):
    #     weight_list.append(model.layers[i].W)
    #     bias_list.append(model.layers[i].b)
    # # print("weight:",weight_list[0][0][0])
    # layers = len(model.layers)
    # encrypted_weight_list, encrypted_bias_list = example_serialize(layers, weight_list, bias_list)
    # weight_dat = encrypted_weight_list
    # bias_dat = encrypted_bias_list
    sema.acquire()
    response_data = pickle.loads(request.data)
    node_address = response_data['node_address']
    if node_address not in ready_to_send:
        ready_to_send.add(node_address)
        if len(weight_data) == 0:
            weight_data = response_data["encrypted_weight_shares"]
            bias_data = response_data["encrypted_bias_shares"]
            # print("recv2",response_data["encrypted_bias_shares"][0:N])
        else:
            weight_data = [x + y for x, y in zip(weight_data, response_data["encrypted_weight_shares"])]
            bias_data = [x + y for x, y in zip(bias_data, response_data["encrypted_bias_shares"])]
            # print("recv3",response_data["encrypted_bias_shares"][0:N])
            # print("2+3",weight_data[0:15])
        # print("data from client", len(response_data["encrypted_weight_shares"]))
    if len(ready_to_send) == len(CONNECTED_NODE_ADDRESS):
        url = "http://127.0.0.1:8000/recv"
        l = 0
        list_w = []
        list_b = []
        for i in range(156):
            for k in range(l,l+5):
                list_w.append(encrypted_weight_list[k])
            l = l + N
        weight_data = [x + y for x, y in zip(weight_data, list_w)]
        # print("recv1",list_w[0:15])
        # print("honest", honest)
        if honest == "no":
            # print("honest Not")
            received_shares = copy(weight_data)
            # print(len(received_shares))
            thr = 3
            for i in range(155):
                indices = random.sample(range(i*5,(i+1)*5), MAX_MISSING + MAX_MANIPULATED)
                missing, manipulated = indices[:MAX_MISSING], indices[MAX_MISSING:]
                for j in missing:     received_shares[j] = None
                for j in manipulated: received_shares[j] = random.randrange(PRIME)
            # print("Corrupted shares: %s" % received_shares)

        l = 0
        for i in range(13):
            for k in range(l,l+5):
                    list_b.append(encrypted_bias_list[k])
            l = l + N
        bias_data = [x + y for x, y in zip(bias_data, list_b)]
        sending_data = {'node_address': myaddress,  'encrypted_weight_shares': weight_data, 'encrypted_bias_shares': bias_data}
        data_byte = pickle.dumps(sending_data)
        print("Length of data byte",len(data_byte))
        requests.post(url,
                    data=data_byte)
    sema.release()
    return 'Data Saved', 200

@app.route('/recv_data_server', methods = ['POST'])
def recv_data_server():
    global data, ready_to_send, s_model, thr, encrypted_weight_list, encrypted_bias_list, weight_list, bias_list, count, model
    recv_data = pickle.loads(request.data)
    recv_weight_shares = recv_data['encrypted_weight_shares']
    # print(recv_weight_shares, len(recv_weight_shares))
    recv_bias_shares = recv_data['encrypted_bias_shares']
    node_address = recv_data['node_address']
    weight_data = []
    bias_data = []
    ready_to_send = set()
    decrypted_weight_list, decrypted_bias_list = example_deserialize(recv_weight_shares, recv_bias_shares)
    count += 1
    for i in range(len(model.layers)):
        model.layers[i].W = decrypted_weight_list[i]
        model.layers[i].b = decrypted_bias_list[i]
    weight_list = []
    bias_list = []
    NN_model, error = training(model)
    if error > thr:
        NN_model, error = training(s_model)
    print("RMSE:", error)
    for i in range(len(NN_model.layers)):
        weight_list.append(NN_model.layers[i].W)
        bias_list.append(NN_model.layers[i].b)
    encrypted_weight_list, encrypted_bias_list = example_serialize(2, weight_list, bias_list)
    # if count != 20 and error < 1:
    #     file1.write(str(error))
    #     file1.write('\n')
    #     file1.close()
    return 'Data Saved', 200