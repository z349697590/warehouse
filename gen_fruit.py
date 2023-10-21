import random

content = None
path = "./durain.txt"
new_path = "./durain_plus.txt"
title = "榴莲"
with open(path, "r", encoding = "utf-8") as f:
    content = f.readlines()

result = []
for i in range(len(content)):
    content[i] = content[i][2:]

num_item = len(content)
k = 1
for i in range(num_item):
    sample_content = random.sample([i for i in range(len(content))], k)
    k += 1
    s = ""
    for j in sample_content:
        s += "它" + content[j][:-2] + ","
    s = s[1:]
    s = s[:-1] + "。"
    s = title + s + "\n"
    print(s)
    result.append(s)

with open(new_path, "w", encoding = "utf-8") as f:
    for res in result:
        f.write(res)

