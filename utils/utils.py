import random

def display_dict(dic, n=10):
    display_key = random.sample(list(dic.keys()), n)
    
    for k in display_key:
        print(k, dic[k])