import os

def txt_write(file_name,str,mode='a'):
    if not os.path.exists(file_name):
        mode = 'w'
    with open(file_name,mode) as f:
        f.write(str)