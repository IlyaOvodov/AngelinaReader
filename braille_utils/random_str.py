"""
Get random files
"""
import random
def not_in(file,used_file):
    for line in used_file:
        if line == file:
            return False
    return True
def randomer(path1,path_used,outp):
#def randomer(path1,outp):
    list = []
    file1 = open(path1)
    used_files = open(path_used) #open fille with used texts
    for line in file1:
        fixline = line #"books\\" + line
        fixline = "books\\" + line
        if not_in(fixline, used_files): #skip alreary used files
            list.append(fixline)
    randlist = random.sample(list, 25)
    outf = open(outp,'w')
    for line in randlist:
        outf.write(line)



if __name__ == "__main__":
    # not texts use train.txt
    already_used = ""
    bookslist = "/home/orwell/brail/results/train_texts.txt"#"/home/orwell/brail/data/AngelinaDataset/books/train.txt"
    #handwrittenlist = "/home/orwell/brail/data/AngelinaDataset/handwritten/train.txt"
    used = "/home/orwell/brail/results/random_iter1.txt" #"/home/orwell/brail/data/AngelinaDataset/train_random_iter1.txt"
    outf = "/home/orwell/brail/data/AngelinaDataset/train_random_iter2_without1.txt"
    randomer(bookslist, used,outf)
    outf = "/home/orwell/brail/data/AngelinaDataset/train_random_iter1_rand2.txt"
    #randomer(bookslist, outf)