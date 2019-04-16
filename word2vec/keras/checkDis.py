import sys
import numpy as np
embedding = sys.argv[1]
word1 = sys.argv[2]
word2 = sys.argv[3]
lineNum = 0
wordCount = 0
sizeDimension = 0
with open(embedding, "r") as input:
    for line in input:
            line = line.rstrip()
            wordCount = int(line.split(" ")[0])
            sizeDimension = int(line.split(" ")[1])
            break
s = (wordCount, sizeDimension)

emb = np.zeros(s, dtype=float)
voc = {}
with open(embedding, "r") as input:
    for line in input:
        if lineNum == 0:
            lineNum += 1
            continue
        else:
            line = line.rstrip()
            word = line.split("\t")[0]
            if word in voc:
                print("error")
            voc[word] = lineNum-1
            embLine = line.split("\t")[1]
            ind = 0
            for i in embLine.split(" "):
                emb[lineNum - 1][ind] = float(i)
                ind += 1
        lineNum += 1
print(emb[voc[word1]].dot(emb[voc[word2]]))


