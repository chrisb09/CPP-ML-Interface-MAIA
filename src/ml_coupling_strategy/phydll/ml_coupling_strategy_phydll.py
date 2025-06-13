from pyphydll.pyphydll import PhyDLL
import mpi4py.MPI as MPI
import time
import sys, random, itertools, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.networks import transformer_tbl, run_encoder_decoder_inference
import deepspeed
import inspect
import math
import argparse

import thoplw

def main():
    ##################
    # SETUP COMM 
    ##################
    dll = PhyDLL()
    dll.init(instance="dl")
    comm = dll.get_local_mpi_comm()
    lrank = comm.Get_rank()   
    globalComm = MPI.COMM_WORLD

    field_count = 1 # number of DL fields
    dll.define_dl(count=field_count)

    dests = dll.get_distribution_info()["dest"]
    num_phy_procs = len(dests)


    ##################
    # Get Meta Information
    ##################
    meta_info_field = []
    for i, dest in enumerate(dests):
        tmp = np.empty(3, dtype=np.int32)
        globalComm.Recv(tmp, source=dest, tag=dest)
        meta_info_field.append(tmp)
    
    #General data where no per process differences occur
    sequence_len = 5
    forecast_window = 2
    checkpoint_path = '/work/thes1961/ai4hpc/checkpoint.pth.tar'    
    cubeD = 8

    #Determine process specific data initially  
    num_cells_per_process = []
    for pid in range(num_phy_procs):
        num_cells_per_process.append(meta_info_field[pid][2])
        print(meta_info_field[pid][2])
    ##################
    # GPU device
    ##################
    num_gpus_per_node = 4 # CLAIX-2023 specific
    # my_device_id is GPU device id if CUDA available, otherwise local rank inside PhyDLL communicator
    if torch.cuda.is_available():
        my_device_id = lrank % num_gpus_per_node
        device = torch.device('cuda', my_device_id)
        torch.cuda.set_device(my_device_id)
    else:
        my_device_id = lrank % comm.Get_size()
        device = torch.device('cpu', my_device_id)
    print(f"Device is {device}")
    print(f"Deviceid is {my_device_id}")

    ##################
    # import DL model
    ##################
    inp_size = cubeD**3
    model = transformer_tbl(
        dim_val=1024,
        inp_dim=inp_size,
        n_dec_lay=6,
        n_enc_lay=6,
        n_heads=16,
        num_pred_features=inp_size,
        batch_first=False
    )

    #Load model checkpoint
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
        print(f"PHYDLL: caught error when loading model: {e}")

    model.to(device)
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
    model.eval()

    ##################
    # Inference
    ##################
    curr_seq_idx = 0
    phy_fields = []
    dl_fields = {"Python-DL-FIELD-OUTPUT": np.zeros(np.cumsum(num_cells_per_process) // sequence_len),}
    while dll.is_phy_signal():
        curr_seq_idx+=1
        
        fields = dll.recv()
        phy_fields.extend(fields["Python-DL-FIELD-INPUT"])

        #Skip the next stuff if not yet gotten all sequences
        if curr_seq_idx < sequence_len:
            continue 

        #Split data per process
        split_points = np.cumsum(num_cells_per_process)[:-1]
        per_proc_flat_data = np.split(phy_fields, split_points)

        #Reshape each chunk into [sequenceLen, vectorLen]
        curr_pos = 0 #Increases by num_cells_per_process[pid] // seqlen
        for pid, flat in enumerate(per_proc_flat_data):
            vectorLen = num_cells_per_process[pid] // sequence_len
            reshaped = flat.reshape(sequence_len, vectorLen)

            inputs = torch.tensor(reshaped, dtype=torch.float32, device=device)
            inputs = inputs.reshape(sequence_len, int(vectorLen / (cubeD**3)), cubeD, cubeD, cubeD)
            inputs = inputs.reshape(*inputs.size()[:-3], -1)

            # Run model inference
            #Input is [sequence_len, num_cubes * 3 * cubeD * cubeD * cubeD]
            with torch.no_grad():
                predictions = run_encoder_decoder_inference(
                    device=device,
                    model=model,
                    src=inputs,
                    forecast_window=forecast_window,
                    batch_size=inputs.shape[1],
                    batch_first=False
                )
                out = predictions[1].view(-1).detach().cpu().numpy()
                dl_fields["Python-DL-FIELD-OUTPUT"][curr_pos:curr_pos+vectorLen] = out
                curr_pos = curr_pos + vectorLen    

        dll.send(dl_fields)

        #Reset data
        phy_fields = []
        curr_seq_idx = 0

    dll.finalize()

if __name__ == "__main__":
    main()