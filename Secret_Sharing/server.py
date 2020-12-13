
import numpy
from flask import Flask, request
import requests, json, pandas, pickle
import requests
import json
import os
import random
import time
import threading

sema = threading.Semaphore()
CONNECTED_NODE_ADDRESS = os.environ.get('peers')
CONNECTED_NODE_ADDRESS = list(CONNECTED_NODE_ADDRESS.split(" "))
len1 = len(CONNECTED_NODE_ADDRESS)
recieved_address = set()
weight_data = []
bias_data = []
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
ini = -1000
B = [1,0,1]
prev_w = []
prev_b = []
prev_h = []
prev_f = []
count = 0
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
fl = 0
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
    # assert(len(points) >= 2*max_error_count + max_degree)
    global ini, prev_f, prev_h, fl ,count
    # interpolate faulty polynomial
    H = lagrange_interpolation(points, values)
    # print("H", H)
    # compute f
    F = [1]
    for xi in points:
        Fi = [base_sub(0, xi), 1]
        F = poly_mul(F, Fi)
    # print("F", F)
    if len(prev_h) == 0:
        sema.acquire()
        prev_h = H
        prev_f = F
        sema.release()
    # run EEA-like algorithm on (F,H) to find EEA triple
    if count == 0:
        R0, R1 = F, H
    else:
        R0, R1 = prev_f, prev_h
        F = prev_f
        H = prev_h
        fl = 1
    S0, S1 = [1], []
    T0, T1 = [], [1]
    while True:
        # print(deg(R0))
        if deg(R0) == max_degree + max_error_count -1:
            count = 1
            print("R0", deg(R0))
            return ini, -1
        else:
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
T = 3
R = T+K
assert(R <= N)

MAX_MISSING = 3
MAX_MANIPULATED = 3
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
    global ini
    points_values = [ (p,v) for p,v in zip(POINTS, shares) if v is not None ]
    # if(len(points_values) >= N - MAX_MISSING):
    #     raise Exception("Too many missing, cannot reconstruct")

    # decode remaining faulty
    points, values = zip(*points_values)
    polynomial, error_locator = gao_decoding(points, values, R, MAX_MANIPULATED)
    if polynomial == ini or polynomial is None:
        return ini, -1
    # check if recovery was possible
    secret = poly_eval(polynomial, 0)
    
    # find error indices
    error_indices = [ i for i,v in enumerate( poly_eval(error_locator, p) for p in POINTS ) if v == 0 ]
    return secret, error_indices

# original_shares0 = shamir_share(5)
# original_shares1 = shamir_share(6)
# original_shares2 = shamir_share(8)
# original_shares4 = shamir_share(7)
# original_shares5 = shamir_share(9)
# original_shares6 = shamir_share(-2)
# original_shares3 = []
# for i in range(15):
#     original_shares3.append(original_shares0[i] + original_shares1[i] + original_shares2[i] + original_shares4[i] + original_shares5[i] + original_shares6[i])

# print("Original shares: %s" % original_shares3)

# introduce faults in shares
def faults(original_shares3):
    received_shares = copy(original_shares3)
    indices = random.sample(range(N), MAX_MISSING + MAX_MANIPULATED)
    missing, manipulated = indices[:MAX_MISSING], indices[MAX_MISSING:]
    for i in missing:     received_shares[i] = None
    for i in manipulated: received_shares[i] = random.randrange(PRIME)
    # print("Received shares: %s" % received_shares)

# # robust reconstruction
# recovered_secret, error_indices = shamir_robust_reconstruct(received_shares)
# print(recovered_secret)
# assert(sorted(error_indices) == sorted(manipulated))

#---------------------------------------------------------------------------------------------------------------------------------------------
#Shamir Completed

def before_sending(weight_list, bias_list):
    rescons_weight = []
    rescons_bias = []
    weight_shares = []
    bias_shares = []
    output_weight = []
    output_bias = []
    global prev_w, prev_b, fl, count
    # print("Aman")
    # print(len(weight_list), weight_list[0][0], weight_list[1][0], weight_list[2][0])
    # print(len(weight_list[0]))
    # print(weight_list[0][2][:15])
    # print(weight_list[0][3][:15])
    weight_list = sorted(weight_list, key=lambda tup: tup[1])
    bias_list = sorted(bias_list, key=lambda tup: tup[1])
    l = 0
    for j in range(156):
        for k in range(l,l+5):
            weight_shares.append(weight_list[0][0][k])
        for k in range(l,l+5):
            weight_shares.append(weight_list[1][0][k])
        for k in range(l,l+5):
            weight_shares.append(weight_list[2][0][k])
        l += 5
    j = 0
    k = 0
    l = 0
    # print(len(bias_list[0][0]))
    for j in range(13):
        for k in range(l,l+5):
            bias_shares.append(bias_list[0][0][k])
        for k in range(l,l+5):
            bias_shares.append(bias_list[1][0][k])
        for k in range(l,l+5):
            bias_shares.append(bias_list[2][0][k])
        l += 5
    # print(bias_shares[0:N])
    l  = 0
    for j in range(156):
        # faults(weight_shares[l:l+N])
        # print("weight_data before stopping", weight_shares[l:l+5])
        recovered_secret, error_indices = shamir_robust_reconstruct(weight_shares[l:l+N])
        if recovered_secret == ini:
            return prev_w, prev_b
        # print("0th element",weight_shares[0])
        l = l + N
        rescons_weight.append(recovered_secret)
    l = 0
    j = 0
    if fl == 1:
        for i in range(156):
            prev_w[i] += numpy.random.normal(0,2,1)[0]

    for j in range(13):
        # faults(bias_shares[l:l+N])
        recovered_secret, error_indices = shamir_robust_reconstruct(bias_shares[l:l+N])
        # print("0th element",bias_shares[0])
        l = l + N
        rescons_bias.append(recovered_secret)
    for i in range(156):
        if rescons_weight[i] > 10000:
            rescons_weight[i] -= PRIME
        rescons_weight[i] = rescons_weight[i]/(len1*1000)
    for i in range(13):
        if rescons_bias[i] > 10000:
            rescons_bias[i] -= PRIME
        rescons_bias[i] = rescons_bias[i]/(len1*1000)
    # print("0th average weight",rescons_weight[0])
    if len(prev_w) == 0:
        sema.acquire()
        prev_w = rescons_weight
        prev_b = rescons_bias
        sema.release()
    # print("reconstructed weight", rescons_weight[0],len(rescons_weight))
    return rescons_weight, rescons_bias


@app.route('/recv', methods = ['POST'])
def recvfromclient():
    global recieved_address,CONNECTED_NODE_ADDRESS, weight_data, bias_data
    recv_data = pickle.loads(request.data)
    node_address = recv_data['node_address']
    if node_address not in recieved_address:
        recieved_address.add(node_address)
        if int(node_address[-1])%len1 == 0 :
            weight_data.append((recv_data['encrypted_weight_shares'],0))
            bias_data.append((recv_data['encrypted_bias_shares'],0))
        elif int(node_address[-1])%len1 == 1:
            weight_data.append((recv_data['encrypted_weight_shares'],1))
            bias_data.append((recv_data['encrypted_bias_shares'],1))
        else:
            weight_data.append((recv_data['encrypted_weight_shares'],2))
            bias_data.append((recv_data['encrypted_bias_shares'],2))
    if len(recieved_address) == len(CONNECTED_NODE_ADDRESS):
        peers = list(CONNECTED_NODE_ADDRESS)
        # i=0
        # print(len(weight_data), len(bias_data))
        sending_weight, sending_bias = before_sending(weight_data, bias_data)
        for peer in peers:
            url = "{}/recv_data_server".format(peer)
            sending_data = {'node_address':'http://127.0.0.1:8000',  'encrypted_weight_shares': sending_weight, 'encrypted_bias_shares': sending_bias}
            data_byte = pickle.dumps(sending_data)
            requests.post(url,
                        data= data_byte)
            print("Broadcasted data after averaging")
        weight_data = []
        bias_data = []
        # print("lengths:", len(weight_data), len(bias_data))
        recieved_address = set()
    return 'Data Saved', 200