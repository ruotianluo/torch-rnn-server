
import os
f = open('arxiv_data.txt', 'w')
for root, dirs, files in os.walk('txt'):
    for fname in files:
        print fname
        tmp = open(os.path.join(root, fname), 'r')
        f.write(tmp.read())
        f.flush()
f.close()
