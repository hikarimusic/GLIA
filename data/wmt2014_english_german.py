import os
import shutil


def download():
    if not os.path.exists(os.path.join(os.getcwd(), "training-parallel-europarl-v7.tgz")):
        os.system("wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz")
        os.system("tar -xvzf training-parallel-europarl-v7.tgz")
    if not os.path.exists(os.path.join(os.getcwd(), "training-parallel-commoncrawl.tgz")):
        os.system("wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz")
        os.system("tar -xvzf training-parallel-commoncrawl.tgz --directory training")
    if not os.path.exists(os.path.join(os.getcwd(), "test-full.tgz")):
        os.system("wget https://www.statmt.org/wmt14/test-full.tgz")
        os.system("tar -xvzf test-full.tgz")
    if not os.path.exists(os.path.join(os.getcwd(), "wmt2014_english_german")):
        os.mkdir(os.path.join(os.getcwd(), "wmt2014_english_german"))
    if not os.path.exists(os.path.join(os.getcwd(), "wmt2014_english_german", "train")):
        os.mkdir(os.path.join(os.getcwd(), "wmt2014_english_german", "train"))
    if not os.path.exists(os.path.join(os.getcwd(), "wmt2014_english_german", "test")): 
        os.mkdir(os.path.join(os.getcwd(), "wmt2014_english_german", "test"))
    
    print("Writing train data - English")
    with open(os.path.join(os.getcwd(), "training", "europarl-v7.de-en.en"), 'r') as f:
        text_1 = f.read()
    with open(os.path.join(os.getcwd(), "training", "commoncrawl.de-en.en"), 'r') as f:
        text_2 = f.read()
    with open(os.path.join(os.getcwd(), "wmt2014_english_german", "train", "english.txt"), 'w') as f:
        f.write(text_1)
        f.write(text_2)
    print("Writing train data - German")
    with open(os.path.join(os.getcwd(), "training", "europarl-v7.de-en.de"), 'r') as f:
        text_1 = f.read()
    with open(os.path.join(os.getcwd(), "training", "commoncrawl.de-en.de"), 'r') as f:
        text_2 = f.read()
    with open(os.path.join(os.getcwd(), "wmt2014_english_german", "train", "german.txt"), 'w') as f:
        f.write(text_1)
        f.write(text_2)

    print("Writing test data - English")
    with open(os.path.join(os.getcwd(), "test-full", "newstest2014-deen-ref.en.sgm"), 'r') as f:
        text = f.readlines()
    f = open(os.path.join(os.getcwd(), "wmt2014_english_german", "test", "english.txt"), 'w')
    for line in filter(lambda s: s[:4]=="<seg", text):
        if line[11] == '>':
            f.write(line[12:-7]+'\n')
        elif line[12] == '>':
            f.write(line[13:-7]+'\n')
        elif line[13] == '>':
            f.write(line[14:-7]+'\n')
    f.close()
    print("Writing test data - German")
    with open(os.path.join(os.getcwd(), "test-full", "newstest2014-deen-ref.de.sgm"), 'r') as f:
        text = f.readlines()
    f = open(os.path.join(os.getcwd(), "wmt2014_english_german", "test", "german.txt"), 'w')
    for line in filter(lambda s: s[:4]=="<seg", text):
        if line[11] == '>':
            f.write(line[12:-7]+'\n')
        elif line[12] == '>':
            f.write(line[13:-7]+'\n')
        elif line[13] == '>':
            f.write(line[14:-7]+'\n')
    f.close()

    os.remove(os.path.join(os.getcwd(), "training-parallel-europarl-v7.tgz"))
    os.remove(os.path.join(os.getcwd(), "training-parallel-commoncrawl.tgz"))
    os.remove(os.path.join(os.getcwd(), "test-full.tgz"))
    shutil.rmtree(os.path.join(os.getcwd(), "training"))
    shutil.rmtree(os.path.join(os.getcwd(), "test-full"))

    print("\n****************************************")
    print("You can download the tokenizers for English and German by the commands below:")
    print("(You will need the tokenizers for lots of nlp tasks.)\n")
    print("python3 -m spacy download en_core_web_sm")
    print("python3 -m spacy download de_core_news_sm")
    print("\n****************************************")



if __name__ == '__main__':
    download()
