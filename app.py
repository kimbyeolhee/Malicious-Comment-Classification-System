import pandas as pd
import numpy as np
import re
import torch
from torch import nn
import gluonnlp as nlp
import emoji
from soynlp.normalizer import repeat_normalize
from googletrans import Translator
from hanspell import spell_checker
from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from flask import*
from flask_ngrok import run_with_ngrok


LABEL_COLUMNS = ["비방", "차별", "성적 수치심 유발", "저주/협박", "악성댓글"]


#입력 댓글 전처리
emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
emojis = ''.join(emojis)
pattern = re.compile(f'[^\x00-\x7Fㄱ-ㅣ가-힣一-龥ぁ-ゔァ-ヴー々〆〤{emojis}]+')
repeatsymbol = re.compile('(([ !"#$%&()*+,-./:;<=>?@[\]^_`{|}~])\\2{1,})') 
url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', str(x)) #한글, 영어, 중국어, 일본어, 이모티콘 외 제거
    for i in range(len(repeatsymbol.findall(x))): #반복 특수문자 축약
        x = x.replace(repeatsymbol.findall(x)[0][0], repeatsymbol.findall(x)[0][1], 1)
    x = url_pattern.sub('', x) #url 제거
    x = x.strip() #앞뒤 공백 제거
    x = repeat_normalize(x, num_repeats=2) #반복 문자 축약
    return x

def translator(reply):
    translator = Translator()
    if translator.detect(reply).lang != 'ko':
        reply = translator.translate(reply, dest="ko").text
    return reply

regexp = re.compile(r'[&]')
def spellchecker(reply):
    reply = spell_checker.check(reply).checked
    return reply


#Kobert, vacab load
bertmodel, vocab  = get_pytorch_kobert_model()


#토크나이저 메서드를 tokenizer에 호출
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)


class BERTClassifier(nn.Module):
    def __init__(self, hidden_size = 768, num_classes = 5, dr_rate = None, params = None):
        
        super(BERTClassifier, self).__init__()
        self.bert = bertmodel
        self.dr_rate = dr_rate
        

        self.classifier = nn.Linear(hidden_size, num_classes)
        
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            
    def generate_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        
        for i,v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.generate_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out=self.dropout(pooler)
        
        return self.classifier(out)
    

#학습 모델 로드
model_save_name = 'test7'
model_file='.pt'
path = f"./{model_save_name}_{model_file}"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTClassifier(num_classes=5, dr_rate = 0.45).to(device)
model.load_state_dict(torch.load(path))

max_len=512
transform = nlp.data.BERTSentenceTransform(tok, max_seq_length = max_len, pad=True, pair=False)


#초기 사용자 데이터
userdata = pd.read_excel('/content/drive/MyDrive/졸업프로젝트/userdata.xlsx')
ID = ''


#captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

PAD_IND = tok.vocab.padding_token
PAD_IND = tok.convert_tokens_to_ids(PAD_IND)
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
lig = LayerIntegratedGradients(model,model.bert.embeddings)

voc_label_dict_inverse={ele:LABEL_COLUMNS.index(ele) for ele in LABEL_COLUMNS}
voc_label_dict={LABEL_COLUMNS.index(ele):ele for ele in LABEL_COLUMNS}

def forward_with_sigmoid_for_bert(input,valid_length,segment_ids):
    return torch.sigmoid(model(input,valid_length,segment_ids))

def forward_for_bert(input,valid_length,segment_ids):
    return torch.nn.functional.softmax(model(input,valid_length,segment_ids),dim=1)

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

def interpret_sentence(model, sentence, min_len = 512, label = 0, n_steps=10):

    # 토크나이징, 시퀀스 생성
    seq_tokens=transform([sentence])
    indexed=torch.tensor(seq_tokens[0]).long()#.to(device)
    valid_length=torch.tensor(seq_tokens[1]).long().unsqueeze(0)
    segment_ids=torch.tensor(seq_tokens[2]).long().unsqueeze(0).to(device)
    sentence=[token for token in tok.sentencepiece(sentence)]
    

    with torch.no_grad():
        model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)
    
    seq_length = min_len

    # predict
    pred = forward_with_sigmoid_for_bert(input_indices,valid_length,segment_ids).detach().cpu().numpy().argmax().item()
    print(forward_with_sigmoid_for_bert(input_indices,valid_length,segment_ids))
    pred_ind = round(pred)
    
    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(input_indices, reference_indices,\
                                           n_steps=n_steps, return_convergence_delta=True,target=label,\
                                           additional_forward_args=(valid_length,segment_ids))

    # print('pred: ', Label.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))

    add_attributions_to_visualizer(attributions_ig, sentence, pred, pred_ind, label, delta, vis_data_records_ig)

def add_attributions_to_visualizer(attributions, input_text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            voc_label_dict[pred_ind], #Label.vocab.itos[pred_ind],
                            voc_label_dict[label], # Label.vocab.itos[label],
                            100, # Label.vocab.itos[1],
                            attributions.sum(),       
                            input_text,
                            delta))

from IPython.core.display import HTML, display
HAS_IPYTHON = True

def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )
def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)

def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 0
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)
#min_len=64 #의미 무엇?
def format_word_importances(words, importances):
    try:
        if importances is None or len(importances) == 0:
            return "<td></td>"

        if len(words) > len(importances):
            words=words[:min_len]


        tags = ["<td>"]
        for word, importance in zip(words, importances[: len(words)]):
            word = format_special_tokens(word)
            color = _get_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                color=color, word=word
            )
            tags.append(unwrapped_tag)
        tags.append("</td>")
        return "".join(tags)
    except Exception as e:
        print("skip it", e)


def visualize_text(datarecords, legend=True):
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
    #    "<th>Word Importance</th>"
    ]

    rows.append(
        "".join(
            [
                "<tr>",
                format_word_importances(
                    datarecords[-1].raw_input_ids, datarecords[-1].word_attributions
                ),
                "<tr>",
            ]
        )
    )


    dom.append("".join(rows))
    dom.append("</table>")
    #display(HTML("".join(dom)))
    dom = str("".join(dom))
    return dom
                          
                            
#flask 실행
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def start_screen():
	return render_template('start_screen.html')


#악성댓글 예측
@app.route('/predict_toxin', methods=['POST'])
def predict_toxin():

    threshold = 0.8

    ID = request.form['input1']
    test_comment = request.form['input2']
    test_comment = clean(test_comment)
    test_comment = translator(test_comment)
    test_comment = spellchecker(test_comment)

    #captum
    interpret_sentence(model, test_comment)
    dom = visualize_text(vis_data_records_ig)

    #사용자 등록
    global userdata
    new = int()
    if userdata[userdata['ID'] == ID].empty:
        new_data = {'ID':ID, '비방':0, '차별':0, '성적 수치심 유발':0, '저주/협박':0, '악성댓글':0, '정상댓글':0}
        userdata = userdata.append(new_data, ignore_index=True)
        new = 1

    #사용자 특성 알림(설정 필요)
    max_troll = 0
    max_columns = 0
    for i in range(1,5):
        if max_troll < int(userdata[userdata.columns[i]][userdata.index[userdata['ID'] == ID]]):
            max_troll = int(userdata[userdata.columns[i]][userdata.index[userdata['ID'] == ID]])
            max_columns = i 
    
    sentences = transform([test_comment])
    get_pred=model(torch.tensor(sentences[0]).long().unsqueeze(0).to(device),torch.tensor(sentences[1]).unsqueeze(0),torch.tensor(sentences[2]).to(device))
    pred=np.array(get_pred.to("cpu").detach().numpy()[0] > threshold, dtype=float)
    pred=np.nonzero(pred)[0].tolist()

    id_index = userdata.index[userdata['ID'] == ID]
    category = ''
    for i in pred:
        if  LABEL_COLUMNS[i] != '악성댓글':
            category += ' ' + LABEL_COLUMNS[i] + ','
            userdata[LABEL_COLUMNS[i]][id_index] = userdata[LABEL_COLUMNS[i]][id_index] + 1
            userdata.to_excel('/content/drive/MyDrive/졸업프로젝트/userdata.xlsx', index=False)
    category = category[:-1]


    #댓글 특성 알림
    if pred == []:
        userdata['정상댓글'][id_index] = userdata['정상댓글'][id_index] + 1
        userdata.to_excel('/content/drive/MyDrive/졸업프로젝트/userdata.xlsx', index=False)
        return render_template('alert2.html', max_troll = max_troll, max_columns = max_columns, ID = ID, new = new, comment = test_comment, dom = dom)

    else:
        userdata['악성댓글'][id_index] = userdata['악성댓글'][id_index] + 1
        userdata.to_excel('/content/drive/MyDrive/졸업프로젝트/userdata.xlsx', index=False)
        return render_template('alert1.html', category = category, max_troll = max_troll, max_columns = max_columns, ID = ID, new = new, comment = test_comment, dom = dom)
        
    
#사용자 삭제
@app.route('/delete_user/<ID>')
def delete_user(ID):
    global userdata
    id_index = userdata.index[userdata['ID'] == ID]
    userdata = userdata.drop(id_index)
    userdata.to_excel('/content/drive/MyDrive/졸업프로젝트/userdata.xlsx', index=False)
    return '%s is deleted!' % escape(ID)


if __name__ == '__main__':
    #app.debug = True
    app.run()
