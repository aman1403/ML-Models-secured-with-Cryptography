
import numpy
import pandas

df = pandas.read_csv('data.csv')

# print(df.info())

# print(df.describe())


# from sklearn.preprocessing import LabelEncoder as LE

# Le = LE()
# types = ("B","M")

# df["diagnosis"] = Le.fit_transform(df["diagnosis"])

# df = df.drop(columns=["id"])

# print(df.columns)

# df.to_csv("data_labelled.csv", index=False)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
y = numpy.array(y)
y = y.reshape((len(y), 1))

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array(X)

print(data_inputs)
# Preparing the NumPy array of the outputs.
data_outputs = y
print(data_outputs)
data_inputs = data_inputs.T
print(data_inputs.shape)
data_outputs = data_outputs.T
print("Modified Data_T", data_inputs)
mean = numpy.mean(data_inputs, axis = 1, keepdims=True)
std_dev = numpy.std(data_inputs, axis = 1, keepdims=True)
data_inputs = (data_inputs - mean)/std_dev
print("Final view", data_inputs)