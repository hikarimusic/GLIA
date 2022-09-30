import os
import shutil

def download():
    if os.path.exists(os.path.join(os.getcwd(), "data")):
        return

    os.mkdir(os.path.join(os.getcwd(), "data"))
    os.mkdir(os.path.join(os.getcwd(), "data", "train"))
    os.mkdir(os.path.join(os.getcwd(), "data", "test"))

    os.system("wget https://www.statmt.org/europarl/v7/fr-en.tgz")
    os.mkdir(os.path.join(os.getcwd(), "train"))
    os.system("tar -xvzf fr-en.tgz --directory train")
    os.system("wget https://www.statmt.org/wmt14/test-full.tgz")
    os.system("tar -xvzf test-full.tgz")

    print("Writing train data - English")
    with open(os.path.join(os.getcwd(), "train", "europarl-v7.fr-en.en"), 'r') as f:
        text = f.read()
    with open(os.path.join(os.getcwd(), "data", "train", "english.txt"), 'w') as f:
        f.write(text)
        
    print("Writing train data - French")
    with open(os.path.join(os.getcwd(), "train", "europarl-v7.fr-en.fr"), 'r') as f:
        text = f.read()
    with open(os.path.join(os.getcwd(), "data", "train", "french.txt"), 'w') as f:
        f.write(text)

    print("Writing test data - English")
    with open(os.path.join(os.getcwd(), "test-full", "newstest2014-fren-ref.en.sgm"), 'r') as f:
        text = f.readlines()
    f = open(os.path.join(os.getcwd(), "data", "test", "english.txt"), 'w')
    for line in filter(lambda s: s[:4]=="<seg", text):
        if line[11] == '>':
            f.write(line[12:-7]+'\n')
        elif line[12] == '>':
            f.write(line[13:-7]+'\n')
        elif line[13] == '>':
            f.write(line[14:-7]+'\n')
    f.close()

    print("Writing test data - French")
    with open(os.path.join(os.getcwd(), "test-full", "newstest2014-fren-ref.fr.sgm"), 'r') as f:
        text = f.readlines()
    f = open(os.path.join(os.getcwd(), "data", "test", "french.txt"), 'w')
    for line in filter(lambda s: s[:4]=="<seg", text):
        if line[11] == '>':
            f.write(line[12:-7]+'\n')
        elif line[12] == '>':
            f.write(line[13:-7]+'\n')
        elif line[13] == '>':
            f.write(line[14:-7]+'\n')
    f.close()

    os.remove(os.path.join(os.getcwd(), "fr-en.tgz"))
    os.remove(os.path.join(os.getcwd(), "test-full.tgz"))
    shutil.rmtree(os.path.join(os.getcwd(), "train"))
    shutil.rmtree(os.path.join(os.getcwd(), "test-full"))


if __name__ == '__main__':
    download()
