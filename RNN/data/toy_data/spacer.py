output = ""
file_name = "toy.test.txt"
string_to_add = " "

with open(file_name, 'r') as f:
    file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]

with open(file_name, 'w') as f:
    f.writelines(file_lines)