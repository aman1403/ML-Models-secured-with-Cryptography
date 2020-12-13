import numpy
from flask import Flask, request
import requests, json, pandas, backprop as bp, pickle
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import tempfile
from pathlib import Path
df = pandas.read_csv('data.csv')
data = df[int((7*len(df))/10):int((8*len(df))/10)]
X = data.drop('charges', axis=1)
y = data['charges']
y = numpy.array(y)
y = y.reshape((len(y), 1))
data_inputs = numpy.array(X)
num_classes = 1
num_inputs = 12
data_outputs = numpy.array(y)
data_inputs = data_inputs.T
data_outputs = data_outputs.T
mean = numpy.mean(data_inputs, axis=1, keepdims=True)
std_dev = numpy.std(data_inputs, axis=1, keepdims=True)
data_inputs = (data_inputs - mean) / std_dev

description = [{"num_nodes" : 12, "activation" : "relu"},
               {"num_nodes" : 1, "activation" : "relu"}]
count = 0
model = bp.NeuralNetwork(description, num_inputs, 'mean_squared', data_inputs, data_outputs, learning_rate=0.001)
app = Flask(__name__)
secure_channel = tempfile.TemporaryDirectory()
sec_con = Path(secure_channel.name)
pk_file = sec_con / 'mypk1.pk'
contx_file = sec_con / 'mycontx1.con'
HE = Pyfhel()
HE.contextGen(p=65537, m=4096)
HE.keyGen()
HE.savepublicKey(pk_file)
HE.saveContext(contx_file)


def example_serialize(layers, weight_list, bias_list):
    encrypted_weight_list = []
    encrypted_bias_list = []
    for i in range(layers):
        for j in range(len(weight_list[i])):
            for k in range(len(weight_list[i][j])):
                encrypted_weight = HE.encryptFrac(weight_list[i][j][k])
                encrypted_weight = encrypted_weight.to_bytes()
                encrypted_weight_list.append(encrypted_weight)

        for a in range(len(bias_list[i])):
            for b in range(len(bias_list[i][a])):
                encrypted_bias = HE.encryptFrac(bias_list[i][a][b])
                encrypted_bias = encrypted_bias.to_bytes()
                encrypted_bias_list.append(encrypted_bias)

    return encrypted_weight_list, encrypted_bias_list


def example_deserialize(weight_list, bias_list):
    decrypted_bias_list = []
    decrypted_weight_list = []
    decrypted_list = [[]]
    i = 0
    cipher = HE.encryptFrac(i)
    #print("De-serializing",len(weight_list))
    while i < len(weight_list) - num_inputs:
        j = 0
        temp_weight = []
        while j < num_inputs:
            weight = weight_list[j]
            cipher.from_bytes(weight, float)
            decrypted_weight = HE.decryptFrac(cipher)
            temp_weight.append(decrypted_weight)
            j = j + 1
        decrypted_list.append(temp_weight)
        i = i + j
    decrypted_weight_list.append(numpy.array(decrypted_list[1:]))
    j = 144
    temp_weight = []
    decrypted_list = [[]]
    while j < len(weight_list):
            weight = weight_list[j]
            cipher.from_bytes(weight, float)
            decrypted_weight = HE.decryptFrac(cipher)
            temp_weight.append(decrypted_weight)
            j = j + 1
    decrypted_list.append(temp_weight)
    decrypted_weight_list.append(numpy.array(decrypted_list[1:]))
    i = 0

    decrypted_list = [[]]
    while i < num_inputs:
        temp_bias = []
        bias = bias_list[i]
        cipher.from_bytes(bias, float)
        decrypted_bias = HE.decryptFrac(cipher)
        temp_bias.append(decrypted_bias)
        decrypted_list.append(temp_bias)
        i = i + 1

    decrypted_bias_list.append(numpy.array(decrypted_list[1:]))
    temp_bias = []
    decrypted_list = [[]]
    bias = bias_list[i]
    cipher.from_bytes(bias, float)
    decrypted_bias = HE.decryptFrac(cipher)
    temp_bias.append(decrypted_bias)
    decrypted_list.append(temp_bias)
    decrypted_bias_list.append(numpy.array(decrypted_list[1:]))
    print("Decryption Done")
    #print(type(decrypted_weight_list[0]), decrypted_bias_list[0])
    return decrypted_weight_list, decrypted_bias_list

def Assign(weight_list, bias_list):
    for i in range(len(model.layers)):
        model.layers[i].W = weight_list[i]
        model.layers[i].b = bias_list[i]

def training(NN_model):
    NN_model.data = data_inputs
    NN_model.labels = data_outputs
    history = NN_model.train(1000)
    prediction = NN_model.layers[(-1)].a
    error = NN_model.calc_accuracy(data_inputs, data_outputs, 'RMSE')
    return (NN_model, error)


@app.route('/', methods=['GET'])
def init():
    global model
    print('client calling server')
    data = {'subject':'handshake',  'node_address':'http://127.0.0.1:5007',  'model':model}
    data_byte = pickle.dumps(data)
    token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJwdWJsaWNfaWQiOiJkZWU0M2JiZC02MzRlLTRlNmUtYWMwYS03MDhkZjliZTQzYmQiLCJleHAiOjE2MDYyNDU0MzJ9.akmEg0IH1-AA46zEIOnURSfBtOuNnfMPxmL5JRwCDsc'
    headers = {'x-access-token': token }
    response = requests.post('http://127.0.0.1:8000/init', data=data_byte,
      headers=headers)
    print(response.status_code)
    return json.dumps({'status': 'Process Started'})


@app.route('/train', methods=['POST'])
def train():
    print('train is called by server')
    global count
    received_data = pickle.loads(request.data)
    if 'subject' in received_data:
        NN_model, error = training(model)
        print(NN_model.layers[0].W)
    else:
        encrypted_weight_list = received_data['weight_list']
        encrypted_bias_list = received_data['bias_list']
        decrypted_weight_list, decrypted_bias_list = example_deserialize(encrypted_weight_list, encrypted_bias_list)
        Assign(decrypted_weight_list, decrypted_bias_list)
        print("Now the training begins")
        NN_model, error = training(model)
    # file1 = open("client7.txt", "a")
    print('Error from model(RMSE) {error}'.format(error=error))
    # if count != 40:
    #     file1.write(str(error))
    #     file1.write('\n')
    #     file1.close()
    weight_list = []
    bias_list = []
    for i in range(len(NN_model.layers)):
        weight_list.append(NN_model.layers[i].W)
        bias_list.append(NN_model.layers[i].b)
    #print(weight_list,bias_list)
    layers = len(NN_model.layers)
    encrypted_weight_list, encrypted_bias_list = example_serialize(layers, weight_list, bias_list)
    #print(len(encrypted_weight_list), len(encrypted_bias_list))
    data = {'weight_list':encrypted_weight_list,  'bias_list':encrypted_bias_list,  'pk_file':pk_file,  'contx_file':contx_file}
    print('Parameters Sent to the server')
    data_byte = pickle.dumps(data)
    #print(len(data_byte))
    return data_byte