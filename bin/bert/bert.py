from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs


config_path = "C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\chinese-bert_chinese_wwm_L-12_H-768_A-12\\publish\\bert_config.json"
checkpoint_path = "C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\chinese-bert_chinese_wwm_L-12_H-768_A-12\\publish\\bert_model.ckpt"
dict_path = "C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\chinese-bert_chinese_wwm_L-12_H-768_A-12\\publish\\vocab.txt"

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)
out = tokenizer.tokenize(u'中办、国办印发《关于做好地方政府专项债券发行及项目配套融资工作的通知》。')
# 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']

print(out)
