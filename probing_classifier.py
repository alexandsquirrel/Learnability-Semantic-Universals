import torch
from transformers import BertTokenizer, BertModel
import h5py
import logging
logging.basicConfig(level=logging.INFO)


# put this method in "quant_verify" to save outputs
def make_hdf5_file(self, out_fn: str):
    """
    Given a list of sentences, tokenize each one and vectorize the tokens. Write the embeddings
    to out_fn in the HDF5 file format. The index in the data corresponds to the sentence index.
    """
    sentence_index = 0
    hidden_rep = []
    with h5py.File(out_fn, 'w') as fout:
        for sentence in hidden_rep:
            try:
                embeddings = self.vectorize(sentence)
            except:
                continue
            fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
            sentence_index += 1


# load (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = []

batch_size = len(texts)

sen_lengths = []

token_texts = []
seg_ids = []
for text in texts:
    t = tokenizer.tokenize(text)
    t[2] = '[MASK]'
    sen_lengths.append(len(t))

    seg_id = []
    for i in range(len(t)):
        seg_id.append(0)

    id = tokenizer.convert_tokens_to_ids(t)
    for i in range(10 - len(t)):
        seg_id.append(0)
        id.append(0)  # padding

    token_texts.append(id)
    seg_ids.append(seg_id)

print(token_texts)
print(seg_ids)



tokenized_text2 = ['[CLS]', 'Trees', '[MASK]', 'not', 'growing', '.', '[SEP]']
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
for i in range(10 - len(indexed_tokens2)):
    indexed_tokens2.append(0)  # padding
segments_ids2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sen_length2 = 7
tokens_tensor2 = torch.tensor([indexed_tokens2])
segments_tensors2 = torch.tensor([segments_ids2])


model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

'''
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')
'''

# Predict hidden states features for each layer
with torch.no_grad():
    output = []
    for i in range(batch_size):
        out = model(torch.tensor([token_texts[i]]), token_type_ids=torch.tensor([seg_ids[i]]))
        output.append(out)

    outputs2 = model(tokens_tensor2, token_type_ids=segments_tensors2)
    encoded_layers2 = outputs2[0]


new_layer0 = encoded_layers2[0]
# print(layer0[0,:])


# Regression for length #

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(768, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x


model_classify = LinearRegression()

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model_classify.parameters(), lr=0.001)

model_classify.train()
for epoch in range(300):
    for i in range(batch_size):
        out = output[i]
        encoded_layers = out[0]
        layer0 = encoded_layers[0]

        x_data = torch.Tensor(layer0[0,:])
        y_data = torch.Tensor([sen_lengths[i]])

        optimizer.zero_grad()
        # Forward pass
        y_pred = model_classify(x_data)
        # Compute Loss
        loss = loss_function(y_pred, y_data)
        # Backward pass
        loss.backward()
        optimizer.step()


###########

print('\n')

new_x = torch.Tensor(new_layer0[0,:])
y_pred = model_classify(new_x)
print("Predicted value: ", y_pred)
