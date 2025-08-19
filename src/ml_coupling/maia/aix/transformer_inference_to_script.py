import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

"""
This code is adapted from AI4HPC to convert its Transformer model into a single Torchscript model.
The advantage is that it only requires the input data and is therefore instantly compatible with the AixeleratorService.
-> The AixService assumes that a model only needs input data and nothing else. This is the easiest way to achieve that, 
no changes to Aix necessary.
The only main difference to the usual python implementation is that the Aixelerator works batch first, which we have to respect here.
"""

########################################
# Positional Encoder
########################################
class PositionalEncoder(nn.Module):
    """
    Positional encoder. Modified for TorchScript compatibility.
    @param seq #tokes at input
    @param encseq_l enclosed part of seq
    @param tgtseq_l target part of seq
    @return parts of sequence cut from
    """
    def __init__(self, dropout: float = 0.1, maxseq_l: int = 5000, d_model: int = 512, batch_first: bool = False):       
        """ Construct
        @param dropout dropout probability
        @param maxseq_l maximum sequence length
        @param d_model
        @param batch_first
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Determine along which dimension the time axis is located.
        self.x_dim = 1 if batch_first else 0

        # Create positional encodings
        position = torch.arange(maxseq_l, dtype=torch.float32).unsqueeze(1)  # shape: [maxseq_l, 1]
        # Use math.log instead of np.log
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(maxseq_l, 1, d_model, dtype=torch.float32)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor with shape [seq, batch, feature] or [batch, seq, feature] 
                              depending on batch_first.
        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        # Add encoding along the time dimension.
        if self.x_dim == 1:
            # self.pe[:x.size(1)] has shape [seq, 1, d_model],
            # so transpose it to [1, seq, d_model] for proper broadcasting.
            pe = self.pe[:x.size(1)].transpose(0, 1)
        else:
            # In the non-batch_first case, the slice already works correctly.
            pe = self.pe[:x.size(0)]
            
        x = x + pe
        
        #x = x + self.pe[:x.size(self.x_dim)]#before adding the above for batchfirst
        return self.dropout(x)
    

########################################
# Transformer Model Definition
########################################
class transformer_tbl(nn.Module):
    """
    Transformer network that expects both encoder and decoder inputs plus masks.
    @return decoder_output
    """
    def __init__(self,
        inp_dim: int,
        batch_first: bool,
        num_pred_features: int,
        dim_val: int = 2048,
        n_enc_lay: int = 6,
        n_dec_lay: int = 6,
        n_heads: int = 8,
        dropout_enc: float = 0.3,
        dropout_dec: float = 0.3,
        dropout_pos_enc: float = 0.1,
        dim_ff_enc: int = 4096,
        dim_ff_dec: int = 4096):
        """ Construct
        @param inp_dim input dimension
        @param batch_first if True, input and output tensors are provided as (batch, seq, feature) instead (seq, batch, feature)
        @param num_pred_features #predicted features
        @param dim_val #expected features in the input
        @param n_enc_lay #sub-encoder-layers in the encoder
        @param n_dec_lay #sub-decoder-layers in the decoder
        @param n_heads #heads in the multiheadattention models
        @param dropout_enc dropout probability for encoder
        @param dropout_dec dropout probability for decoder
        @param dropout_pos_enc dropout probability for positional encoder
        @param dim_ff_enc dimension of the feedforward network model, encoder side
        @param dim_ff_dec dimension of the feedforward network model, decoder side
        @return parts of sequence cut from
        """
        super().__init__()
        self.batch_first = batch_first

        # Input linear mappings
        self.enc_inp = nn.Linear(inp_dim, dim_val)
        self.dec_inp = nn.Linear(num_pred_features, dim_val)
        self.linear_map = nn.Linear(dim_val, num_pred_features)
        # Activation
        self.tanh = nn.Tanh()

        # Positional Encoder
        self.positional_enc_layer = PositionalEncoder(dropout=dropout_pos_enc,
                                                      d_model=dim_val,
                                                      batch_first=batch_first)
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_ff_enc,
            dropout=dropout_enc,
            batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=n_enc_lay,
            norm=nn.LayerNorm(dim_val))
        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_ff_dec,
            dropout=dropout_dec,
            batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_dec_lay,
            norm=nn.LayerNorm(dim_val))

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masks.
        Args:
            src (torch.Tensor): Source tensor.
            tgt (torch.Tensor): Target tensor.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.
            
        Returns:
            torch.Tensor: The linear-mapped decoder output.
        """
        src = self.enc_inp(src)
        src = self.positional_enc_layer(src)
        src = self.encoder(src)
        decoder_output = self.dec_inp(tgt)
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask)
            #memory_mask=src_mask)#Also like this in ai4hpc
        return self.linear_map(decoder_output)
        # If you want to apply tanh activation, you can return: #Not done in AI4HPC
        # return self.tanh(self.linear_map(decoder_output))


########################################
# Helper function for mask generation
########################################
def generate_square_subsequent_mask(device: torch.device, dim1: int, dim2: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on the diagonal.
    
    Args:
        device (torch.device): The device to place the mask.
        dim1 (int): First dimension.
        dim2 (int): Second dimension.
        
    Returns:
        torch.Tensor: The generated mask.
    """
    mask = torch.triu(torch.ones(dim1, dim2, dtype=torch.float32) * float('-inf'), diagonal=1)
    return mask.to(device)


########################################
# Inference Wrapper Module
########################################
class TransformerInferenceWrapper(nn.Module):
    """
    A TorchScript-friendly wrapper that encapsulates the forecasting loop.
    It takes only the source tensor (src) as input and iteratively generates the target sequence.
    """
    def __init__(self, model: transformer_tbl, forecast_window: int, batch_first: bool):
        super().__init__()
        self.model = model
        self.forecast_window = forecast_window  # Total number of prediction steps (>= 1)
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Given only src as input, the forward loop builds an initial tgt and performs forecasting. 
        Slight semantic changes from Ai4HPC, but same functionality.

        Args:
            src (torch.Tensor): Source input tensor.
            
        Returns:
            torch.Tensor: Final prediction output.
        """
        device = input.device
        # Initialize tgt using the last time step of src,
        # ensuring the target sequence has the correct shape.
        if self.batch_first:
            # src shape: [batch, seq, features] --> tgt shape: [batch, 1, features]
            tgt = input[:, -2:, :] #note that in this and the other calls where unsqueeze was previously done we dont have to because we added an : behind the -1
            src = input[:, :-2, :]
            seq_dim = 1
        else:
            # src shape: [seq, batch, features] --> tgt shape: [1, batch, features]
            tgt = input[-2:, :, :]
            src = input[:-2, :, :]
            seq_dim = 0

        # Forecast loop: forecast_window-1 iterative steps appended to tgt.
        for _ in range(self.forecast_window - 1):
            if self.batch_first:
                dim_a = tgt.size(1)
                dim_b = src.size(1)
            else:
                dim_a = tgt.size(0)
                dim_b = src.size(0)

            tgt_mask = generate_square_subsequent_mask(device, dim_a, dim_a)
            src_mask = generate_square_subsequent_mask(device, dim_a, dim_b)

            # Generate prediction from the model
            prediction = self.model(src, tgt, src_mask, tgt_mask)

            # Extract the last predicted time step and unsqueeze to maintain sequence dimension.
            if self.batch_first:
                last_predicted = prediction[:, -1:, :]
            else:
                last_predicted = prediction[-1:, :, :]

            # Append new prediction to the target sequence along the time dimension.
            tgt = torch.cat((tgt, last_predicted), dim=seq_dim)

        # Final prediction with the full target sequence.
        if self.batch_first:
            dim_a = tgt.size(1)
            dim_b = src.size(1)
        else:
            dim_a = tgt.size(0)
            dim_b = src.size(0)
        tgt_mask = generate_square_subsequent_mask(device, dim_a, dim_a)
        src_mask = generate_square_subsequent_mask(device, dim_a, dim_b)
        return self.model(src, tgt, src_mask, tgt_mask)


########################################
# Script and Save the Entire Pipeline
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--cube-dim", help="Size of cube per dimension", type=int, default=8)
    parser.add_argument("--forecast-window", help="forecast_window", type=int, default=2)
    parser.add_argument("--checkpoint-path", help="Checkpoint path", type=str, default='/work/thes1961/ai4hpc/checkpoint.pth.tar')

    # Parse the arguments
    # Define your hyperparameters (adjust these as needed)
    args = parser.parse_args()
    cubeD = args.cube_dim
    forecast_window = args.forecast_window
    checkpoint_path = args.checkpoint_path    


    inp_size = cubeD**3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_device_id = 0  # replace with your device ID if needed

    # Instantiate the transformer using the same parameters as when training.
    model = transformer_tbl(
        dim_val=1024,
        inp_dim=inp_size,
        n_enc_lay=6,
        n_dec_lay=6,
        n_heads=16,
        num_pred_features=inp_size,
        batch_first=True  # Use the same value as in training
    )

    # Load your checkpoint (assuming checkpoint.pth.tar contains a key 'state_dict')
    loc = {'cuda:%d' % 0: 'cuda:%d' % my_device_id} if torch.cuda.is_available() else "cpu"
    print(f"Location is {loc}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        new_state_dict = {}
        for k in checkpoint['state_dict']:
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = checkpoint['state_dict'][k]
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    model.to(device)
    torch.set_float32_matmul_precision('high')
    model.eval()

    # Create the inference wrapper that accepts only src as input.
    inference_wrapper = TransformerInferenceWrapper(model, forecast_window, batch_first=True)
    inference_wrapper.eval()

    # Script the entire pipeline (model + forecasting loop).
    scripted_inference = torch.jit.script(inference_wrapper)
    scripted_inference.save("transformer_inference_scripted.pt")
    print("TorchScript inference model saved as transformer_inference_scripted.pt")