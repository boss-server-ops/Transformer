import json
import os

def load_corpus(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_corpus(file_path, lines):
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def process_corpus(files, ch_path, en_path):
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = load_corpus(os.path.join('./json', file + '.json'))
        for item in corpus:
            ch_lines.append(item[1] + '\n')
            en_lines.append(item[0] + '\n')

    write_corpus(ch_path, ch_lines)
    write_corpus(en_path, en_lines)

    print("lines of Chinese: ", len(ch_lines))
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")

if __name__ == "__main__":
    files = ['train', 'dev', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    process_corpus(files, ch_path, en_path)