import sentencepiece as spm

class SentencePieceTrainer:
    def __init__(self, input_file, vocab_size, model_name, model_type, character_coverage):
        self.input_file = input_file
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.model_type = model_type
        self.character_coverage = character_coverage

    def train(self):
        input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                         '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        cmd = input_argument % (self.input_file, self.model_name, self.vocab_size, self.model_type, self.character_coverage)
        spm.SentencePieceTrainer.Train(cmd)

class SentencePieceTester:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def test(self, text):
        print(self.sp.EncodeAsPieces(text))
        print(self.sp.EncodeAsIds(text))
        a = [12907, 277, 7419, 7318, 18384, 28724]
        print(self.sp.decode_ids(a))

if __name__ == "__main__":
    en_trainer = SentencePieceTrainer('../data/corpus.en', 32000, 'eng', 'bpe', 1)
    en_trainer.train()

    ch_trainer = SentencePieceTrainer('../data/corpus.ch', 32000, 'chn', 'bpe', 0.9995)
    ch_trainer.train()

    tester = SentencePieceTester('./chn.model')
    tester.test("美国总统特朗普今日抵达夏威夷。")