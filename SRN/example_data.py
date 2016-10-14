import pickle

s1 = 'ABC'
s2 = 'AFC'
s3 = 'DBE'
s4 = 'DGE'

unique = sorted(list(set([char for char in s1+s2+s3+s4])))
print(unique)
example_data = [s1,s2,s3,s4,unique]

pickle.dump(example_data, open('a_to_g.pkl', 'wb'))