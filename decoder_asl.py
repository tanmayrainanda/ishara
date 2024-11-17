import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_decoder_layers, num_classes, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size

        # Transformer decoder layer configuration
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # Fully Connected layers for output
        self.fc_confidence = nn.Linear(embed_size, 1)  # For confidence score
        self.fc_class_output = nn.Linear(embed_size, num_classes)  # For class output

    def forward(self, encoder_output, tgt):
        # tgt is the input to the decoder
        # encoder_output is the output from the encoder
        
        # Transformer decoder processing
        decoder_output = self.transformer_decoder(tgt, encoder_output)
        
        # Confidence output
        confidence_output = torch.sigmoid(self.fc_confidence(decoder_output))
        
        # Class prediction output
        class_output = self.fc_class_output(decoder_output)
        
        return confidence_output, class_output