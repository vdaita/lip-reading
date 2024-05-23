import torch
import torch.nn.functional as F
import math

class ImageEmbeddings(torch.nn.Module):
    def __init__(self, dropout_value=0.1, n_embed=3, max_seq_len=512):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(max_seq_len, n_embed)

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool1 = torch.nn.MaxPool2d(5, 5)
        self.dropout1 = torch.nn.Dropout2d(dropout_value)
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.pool2 = torch.nn.MaxPool2d(5, 5)
        self.dropout2 = torch.nn.Dropout(dropout_value)
        self.fc1 = torch.nn.Linear(12 * 5 * 5, 120)
        self.dropout3 = torch.nn.Dropout(dropout_value)
        self.fc2 = torch.nn.Linear(120, n_embed)
        self.layer_norm = torch.nn.LayerNorm(n_embed)
        self.dropout4 = torch.nn.Dropout(dropout_value)

    def forward(self, x):
        # Given an image, process the image and reflect some information about the image. 
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x += self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout4(x)

        return x

class BertTextEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, n_embed=3, max_seq_len=16):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_embed = n_embed

        self.word_embeddings = torch.nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = torch.nn.Embedding(max_seq_len, n_embed)
        self.token_type_embeddings = torch.nn.Embedding(2, n_embed)

        self.LayerNorm = torch.nn.LayerNorm(n_embed, eps=1e-12, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)
        words_embeddings = self.word_embeddings(x)
        position_embeddings = self.position_embeddings(position_ids.unsqueeze(0).repeat(x.shape[0], -1, -1))
        segments_embeddings = self.token_type_embeddings(torch.tensor([1] * x.shape[0])) # Segment is guaranteed to be 1 because all text is generated. Repeat 1, B times. 

        print("Words embeddings: ", words_embeddings.shape)
        print("Position embeddings: ", position_embeddings.shape)
        print("Segments embeddings: ", segments_embeddings.shape)

        embeddings = words_embeddings + position_embeddings + segments_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertAttentionHead(torch.nn.Module):
    """
    A single attention head in MultiHeaded Self Attention layer.
    The idea is identical to the original paper ("Attention is all you need"),
    however instead of implementing multiple heads to be evaluated in parallel we matrix multiplication,
    separated in a distinct class for easier and clearer interpretability
    """

    def __init__(self, head_size, dropout=0.1, n_embed=3):
        super().__init__()

        self.query = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(in_features=n_embed, out_features=head_size)

        self.dropout_value = dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # B, Seq_len, N_embed
        B, seq_len, n_embed = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(n_embed)  # (B, Seq_len, Seq_len)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask out not attended tokens

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class BertSelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        head_size = n_embed # // n_heads

        n_heads = n_heads

        self.heads = torch.nn.ModuleList([BertAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])

        # self.proj = torch.nn.Linear(head_size * n_heads, n_embed)  # project from multiple heads to the single space
        self.proj = torch.nn.Linear(head_size, n_embed)  # project from multiple heads to the single space

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        context = torch.cat([head(x, mask) for head in self.heads], dim=-1)

        proj = self.proj(context)

        out = self.dropout(proj)

        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffwd(x)

        return out


class BertLayer(torch.nn.Module):
    """
    Single layer of BERT transformer model
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = BertSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, x, mask): # Not using residual connections in paper
        x = self.layer_norm1(x + self.self_attention(x, mask))
        out = self.layer_norm2(x + self.feed_forward(x))
        return out


class BertEncoder(torch.nn.Module):
    def __init__(self, n_layers=2, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        self.layers = torch.nn.ModuleList([BertLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertDecoder(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3, n_tokens=30522):
        super().__init__()

        self.dense1 = torch.nn.Linear(in_features=n_embed, out_features=n_embed)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(n_embed)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.dense2 = torch.nn.Linear(in_features=n_embed, out_features=n_tokens)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.layer_norm(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        out = self.activation(x)

        return out

class NanoBERT(torch.nn.Module):
    """
    NanoBERT is a almost an exact copy of a transformer decoder part described in the paper "Attention is all you need"
    This is a base model that can be used for various purposes such as Masked Language Modelling, Classification,
    Or any other kind of NLP tasks.
    This implementation does not cover the Seq2Seq problem, but can be easily extended to that.
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        """

        :param vocab_size: size of the vocabulary that tokenizer is using
        :param n_layers: number of BERT layer in the model (default=2)
        :param n_heads: number of heads in the MultiHeaded Self Attention Mechanism (default=1)
        :param dropout: hidden dropout of the BERT model (default=0.1)
        :param n_embed: hidden embeddings dimensionality (default=3)
        :param max_seq_len: max length of the input sequence (default=16)
        """
        super().__init__()

        self.embedding = BertTextEmbeddings(vocab_size, n_embed, max_seq_len//2)
        self.image_embeddings = ImageEmbeddings(n_embed=n_embed, max_seq_len=max_seq_len//2)

        self.encoder = BertEncoder(n_layers, n_heads, dropout, n_embed)

        self.decoder = BertDecoder(dropout, n_embed)

    def forward(self, images, x, targets=None):
        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
        text_embeddings = self.embedding(x)
        image_embeddings = self.image_embeddings(images)
        embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)

        encoded = self.encoder(embeddings, mask)
        decoded = self.decoder(encoded)

        if targets == None:
            return decoded, None
        else:
            return decoded, F.cross_entropy(decoded, targets)
        
tiny_bert_mapping = {
        "bert.embeddings.word_embeddings.weight": "embedding.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight": "embedding.position_embeddings.weight",
        "bert.embeddings.token_type_embeddings.weight": "embedding.token_type_embeddings.weight",
        "bert.embeddings.LayerNorm.weight": "embedding.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias": "embedding.LayerNorm.bias",
        "bert.encoder.layer.0.attention.self.query.weight": "encoder.layers.0.self_attention.heads.0.query.weight",
        "bert.encoder.layer.0.attention.self.query.bias": "encoder.layers.0.self_attention.heads.0.query.bias",
        "bert.encoder.layer.0.attention.self.key.weight": "encoder.layers.0.self_attention.heads.0.key.weight",
        "bert.encoder.layer.0.attention.self.key.bias": "encoder.layers.0.self_attention.heads.0.key.bias",
        "bert.encoder.layer.0.attention.self.value.weight": "encoder.layers.0.self_attention.heads.0.values.weight",
        "bert.encoder.layer.0.attention.self.value.bias": "encoder.layers.0.self_attention.heads.0.values.bias",
        "bert.encoder.layer.0.attention.output.dense.weight": "encoder.layers.0.self_attention.proj.weight",
        "bert.encoder.layer.0.attention.output.dense.bias": "encoder.layers.0.self_attention.proj.bias",
        "bert.encoder.layer.0.attention.output.LayerNorm.weight": "encoder.layers.0.layer_norm2.weight",
        "bert.encoder.layer.0.attention.output.LayerNorm.bias": "encoder.layers.0.layer_norm2.bias",
        "bert.encoder.layer.0.intermediate.dense.weight": "encoder.layers.0.feed_forward.ffwd.0.weight",
        "bert.encoder.layer.0.intermediate.dense.bias": "encoder.layers.0.feed_forward.ffwd.0.bias",
        "bert.encoder.layer.0.output.dense.weight": "encoder.layers.0.feed_forward.ffwd.2.weight",
        "bert.encoder.layer.0.output.dense.bias": "encoder.layers.0.feed_forward.ffwd.2.bias",
        "bert.encoder.layer.0.output.LayerNorm.weight": "encoder.layers.0.layer_norm1.weight",
        "bert.encoder.layer.0.output.LayerNorm.bias": "encoder.layers.0.layer_norm1.bias",
        "bert.encoder.layer.1.attention.self.query.weight": "encoder.layers.1.self_attention.heads.0.query.weight",
        "bert.encoder.layer.1.attention.self.query.bias": "encoder.layers.1.self_attention.heads.0.query.bias",
        "bert.encoder.layer.1.attention.self.key.weight": "encoder.layers.1.self_attention.heads.0.key.weight",
        "bert.encoder.layer.1.attention.self.key.bias": "encoder.layers.1.self_attention.heads.0.key.bias",
        "bert.encoder.layer.1.attention.self.value.weight": "encoder.layers.1.self_attention.heads.0.values.weight",
        "bert.encoder.layer.1.attention.self.value.bias": "encoder.layers.1.self_attention.heads.0.values.bias",
        "bert.encoder.layer.1.attention.output.dense.weight": "encoder.layers.1.self_attention.proj.weight",
        "bert.encoder.layer.1.attention.output.dense.bias": "encoder.layers.1.self_attention.proj.bias",
        "bert.encoder.layer.1.attention.output.LayerNorm.weight": "encoder.layers.1.layer_norm2.weight",
        "bert.encoder.layer.1.attention.output.LayerNorm.bias": "encoder.layers.1.layer_norm2.bias",
        "bert.encoder.layer.1.intermediate.dense.weight": "encoder.layers.1.feed_forward.ffwd.0.weight",
        "bert.encoder.layer.1.intermediate.dense.bias": "encoder.layers.1.feed_forward.ffwd.0.bias",
        "bert.encoder.layer.1.output.dense.weight": "encoder.layers.1.feed_forward.ffwd.2.weight",
        "bert.encoder.layer.1.output.dense.bias": "encoder.layers.1.feed_forward.ffwd.2.bias",
        "bert.encoder.layer.1.output.LayerNorm.weight": "encoder.layers.1.layer_norm1.weight",
        "bert.encoder.layer.1.output.LayerNorm.bias": "encoder.layers.1.layer_norm1.bias",
        "cls.predictions.bias": None,
        "cls.predictions.transform.dense.weight": None, 
        "cls.predictions.transform.dense.bias": None,
        "cls.predictions.transform.LayerNorm.weight": None,
        "cls.predictions.transform.LayerNorm.bias": None,
        "cls.predictions.decoder.weight": None,
        "cls.predictions.decoder.bias": None
    }

def load_pretrained_bert_model():
    tiny_bert = NanoBERT(30522, n_heads=2, n_layers=2, n_embed=128, max_seq_len=512)
    from transformers import BertLMHeadModel
    import json
    model_hf = BertLMHeadModel.from_pretrained("prajjwal1/bert-tiny")
    sd_hf = model_hf.state_dict()
    sd = tiny_bert.state_dict()

    layer_names_hf = list(sd_hf)
    layer_names = list(sd)

    print("HF Model Layer names: ")
    print(json.dumps(layer_names_hf, indent=4))
    print("Local Layer Names: ")
    print(json.dumps(layer_names, indent=4))

    for layer_name in layer_names_hf:
        print(layer_name)
        print("Shape: ", sd_hf[layer_name].shape)
    
    print("------")

    for layer_name in layer_names_hf:
        mapped_name = tiny_bert_mapping[layer_name]
        if mapped_name:
            if "position" in mapped_name:
                # need to chop this layer in half
                print(layer_name)
                print("Yours: ", sd[mapped_name].shape)
                print("Theirs: ", sd_hf[layer_name].shape)
                print("Will only select the first half to transfer over")
                print("=====")

                hf_layer = sd_hf[layer_name]
                max_value = hf_layer.shape[0]//2
                cropped_layer = hf_layer[:max_value,:]
                print(cropped_layer.shape)

                assert sd[mapped_name].shape == cropped_layer.shape
                sd[mapped_name].copy_(cropped_layer)
            else:
                print(layer_name)
                print("Yours: ", sd[mapped_name].shape)
                print("Theirs: ", sd_hf[layer_name].shape)
                print("=====")
                assert sd[mapped_name].shape == sd_hf[layer_name].shape
                sd[mapped_name].copy_(sd_hf[layer_name])
    tiny_bert.load_state_dict(sd)
    return tiny_bert
    
if __name__ == "__main__":
    load_pretrained_bert_model()