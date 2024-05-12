from flask import Flask, request, jsonify,render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizerFast
import torch.nn as nn
from transformers import BertConfig, BertModel

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained('roberta-base',add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained('rishieee/robertaCwTest', num_labels=4)
 # Assuming binary classification
#Mirko

# Load the fine-tuned model weights
#model_path = 'biobert_postags_cased__e_16.pt'
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
'''
class BioBertPosTagsClassifier(nn.Module):
  def __init__(self, output_dim, pos_vocab_size, pos_embedding_dim):
    super(BioBertPosTagsClassifier, self).__init__()
    self.model_label = 'biobert_postags_cased'
    self.bert = BertModel.from_pretrained('dmis-lab/biobert-v1.1',
                                          num_labels=output_dim,
                                          add_pooling_layer=False)

    # Add 1 to pos_vocab_size to account for the special -100 index.
    # We'll reserve the last embedding vector for -100 indices.
    self.pos_embedding = nn.Embedding(num_embeddings=pos_vocab_size + 1,
                                      embedding_dim=pos_embedding_dim,
                                      padding_idx=pos_vocab_size)

    # Adjust the input size of the classifier
    combined_embedding_dim = self.bert.config.hidden_size + pos_embedding_dim
    self.fc = nn.Linear(combined_embedding_dim, output_dim)

  def forward(self, text, attention_mask, pos_tags):
    outputs = self.bert(text, attention_mask=attention_mask, return_dict=False)

    sequence_output = outputs[0]  # [batch_size, sequence_length, 768]

    # Adjust pos_tags to ensure -100 indices map to the last embedding vector
    adjusted_pos_tags = torch.where(pos_tags == -100, torch.tensor(self.pos_embedding.padding_idx, device=pos_tags.device), pos_tags)

    # Get embeddings from POS tags
    pos_embeddings = self.pos_embedding(adjusted_pos_tags)

    # Concatenate BERT and POS embeddings
    combined_embeddings = torch.cat((sequence_output, pos_embeddings), dim=-1)

    logits = self.fc(combined_embeddings)

    return logits
output_dim = 4
pos_vocab_size = 18
pos_embedding_dim = 16

biobertpostags_model = BioBertPosTagsClassifier(output_dim, pos_vocab_size, pos_embedding_dim)
model_path = 'biobert_postags_cased__e_16.pt'
biobertpostags_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
'''
#model.eval()
label_list_rob = ['B-O', 'B-AC', 'B-LF', 'I-LF']
# Define a function for tokenizing and preprocessing text
#check_point = 'dmis-lab/biobert-v1.1'
#tokenizer = BertTokenizerFast.from_pretrained(model)
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# Define a function to make predictions
def predict_sequence_classification(text):
    inputs = preprocess_text(text)
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=2)#.squeeze().tolist()
    print(predicted_classes)
    predict_final = predicted_classes.tolist()[0]
    print(predict_final)
    #predicted_labels = [label_list_rob[label_id] for label_id in predicted_classes]
    return predict_final

# Define a route for rendering the HTML form
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sequence Classification</title>
    </head>
    <body>
        <h1>Sequence Classification</h1>
        <form action="/predict" method="post">
            <label for="text">Enter Text:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    '''

    
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
     #text = request.form['text']
     #To use postman
    data= request.get_json()
    text=data['text']
    #to use html local
    predictions = predict_sequence_classification(text)
    
    # Render the predicted labels in the results.html template
    return jsonify({"predictions":predictions,"input_text":text})
    #return render_template('result.html',input_text=text, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
