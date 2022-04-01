"""
Get random files
"""
import random
def randomer(path1,path2,outp):
    list = []
    file1 = open(path1)
    for line in file1:
        fixline = "books\\" + line
        list.append(fixline)
    file2 = open(path2)
    for line in file2:
        fixline = "handwritten\\" + line
        list.append(fixline)
    randlist = random.sample(list, 40)
    outf = open(outp,'w')
    for line in randlist:
        outf.write(line)



if __name__ == "__main__":
    bookslist = "/home/orwell/brail/data/AngelinaDataset/books/train.txt"
    handwrittenlist = "/home/orwell/brail/data/AngelinaDataset/handwritten/train.txt"
    outf = "/home/orwell/brail/data/AngelinaDataset/train_random_iter1.txt"
    randomer(bookslist, handwrittenlist,outf)