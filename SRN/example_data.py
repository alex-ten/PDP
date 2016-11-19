import pickle
from SRN.FSM import DataSet

s1 = 'ABCDEFX'
s2 = 'ABCX'
s3 = 'ADEFX'

unique = sorted(list(set([char for char in s1+s2+s3])))
print(unique)
example_data = [s1,s2,s3,unique]

pickle.dump(example_data, open('a_through_x.pkl', 'wb'))