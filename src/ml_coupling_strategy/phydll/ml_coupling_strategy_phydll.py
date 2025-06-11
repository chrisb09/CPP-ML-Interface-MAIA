from pyphydll.pyphydll import PhyDLL
import mpi4py.MPI as MPI
import time
import sys, random, itertools, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from src.networks import transformer_tbl, run_encoder_decoder_inference
import deepspeed
import inspect
import math
import argparse

import thoplw

def main():
    ##################
    # Create & use input arguments 
    ##################
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--seq-len", help="Input sequence length", type=int, default=5)
    parser.add_argument("--output-len", help="Output sequence length", type=int, default=2)
    parser.add_argument("--checkpoint", help="Checkpoint path", type=str, default='/work/thes1961/ai4hpc/checkpoint.pth.tar')

    # Parse the arguments
    args = parser.parse_args()
    sequence_len = args.seq_len
    forecast_window = args.output_len
    checkpoint_path = args.checkpoint    
    
    ##################
    # SETUP COMM 
    ##################
    print(f"PHYDLL TEST 1 before PhyDLL init", flush=True)
    dll = PhyDLL()
    dll.init(instance="dl")
    comm = dll.get_local_mpi_comm()
    lrank = comm.Get_rank()   
    globalComm = MPI.COMM_WORLD
    print(f"PHYDLL TEST 2 after PhyDLL init, local rank = {lrank}", flush=True)
    print(f"PHYDLL TEST 3 after PhyDLL init, globalComm rank = {globalComm.Get_rank()}", flush=True)
    ##################
    # Setup Fields
    ##################
    field_count = 1 # number of DL fields

    dll.define_dl(count=field_count)
    
    phy_count, dl_count = dll.get_field_counts()
    #print(f"PHYDLL: number of physical fields = {phy_count}, DL fields = {dl_count}")

    dests = dll.get_distribution_info()["dest"]
    print(f"PHYDLL: destination ranks = {dests}", flush=True)
    num_phy_procs = len(dests)
    #print(f"PHYDLL ({lrank}): number of physical procs = {num_phy_procs}")

    cubes_tensors = [None] * sequence_len
    phy_fields = {} # The fields we get 
    dl_fields = { # The fields we send later
        "Python-DL-FIELD-0": np.zeros(dll.get_field_size()),
        "Python-DL-FIELD-1": np.zeros(dll.get_field_size()),
        "Python-DL-FIELD-2": np.zeros(dll.get_field_size()),
    }
    cubes_per_process, field_shape_per_process = {},  [None] * num_phy_procs
    #print(f"PHYDLL: dll.get_field_size() = {dll.get_field_size()}")
    meta_info_field = []
    print(f"{globalComm.Get_size()} processes in total", flush=True)
    ##################
    # Get Meta Information
    ##################
    print("Getting meta information from physical processes", flush=True)
    #requests = []
    for i, dest in enumerate(dests):
        print(f"PHYDLL: requesting meta info from process {dest}", flush=True)
        tmp = np.empty(7, dtype=np.int32)
        globalComm.Recv(tmp, source=dest, tag=dest)
        meta_info_field.append(tmp)
        #requests.append(req)
        print(f"PHYDLL: Appended request", flush=True)
    #MPI.Request.waitall(requests)
    print(f"PHYDLL: meta_info_field = {meta_info_field}", flush=True)

    #Determine process specific data initially  
    num_cells_per_process = []
    for pid in range(num_phy_procs):
        cubes_per_process[pid] = [None] * sequence_len
        field_shape_per_process[pid] = (meta_info_field[pid][0], meta_info_field[pid][1], meta_info_field[pid][2])
        num_cells_per_process.append(math.prod(field_shape_per_process[pid]))
        #print(f"PHYDLL: cells of process = {num_cells_per_process[pid]}")

    ##################
    # Cube parameters
    ##################
    cubeC = 2 # monotonic cuts
    cubeD = 8 # cube dimension = 8^3

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

    displacements_per_process = [0]
    for pid in range(1, num_phy_procs):
        displacements_per_process.append(
            displacements_per_process[-1] + num_cells_per_process[pid - 1]
        )

    count = 0
    flat = []
    while dll.is_phy_signal():
        count+=1
        fields = dll.recv()
        print(list(fields.keys()))
        print(fields)
        flat.extend(fields["Python-DL-FIELD-INPUT"])
        if count < sequence_len:
            continue 
        print(flat)
        print(flat.shape)
        print(sequence_len )
        print(num_cells_per_process)
        print(num_phy_procs)
        # len(flat) == sum(num_cells_per_process)

        offset = 0
        #for pid in range(num_phy_procs):
        #assert len(flat) == sequence_len * num_cells_per_process * num_phy_procs


        # Step 2: Split data per process
        split_points = np.cumsum(num_cells_per_process)[:-1]
        per_proc_flat_data = np.split(np.array(flat), split_points)

        # Step 3: Reshape each chunk into [sequenceLen, vectorLen]
        #input_fields_per_proc = []
        curr_pos = 0
        for pid, flat in enumerate(per_proc_flat_data):
            vectorLen = num_cells_per_process[pid] // sequence_len
            reshaped = flat.reshape(sequence_len, vectorLen)
            #input_fields_per_proc.append(reshaped)
            print(reshaped.shape)
            print(reshaped.shape)

            inputs = torch.tensor(flat, dtype=torch.float32, device=device).reshape(sequence_len, num_cells_per_process * num_phy_procs)
            print(inputs.shape)
            #inputs = torch.stack(cubes_tensors[seq_idx+1:] + cubes_tensors[:seq_idx+1])
            inputs = inputs.reshape(sequence_len, inputs.shape[1]*inputs.shape[2], cubeD, cubeD, cubeD)
            print(inputs.shape)
            inputs = inputs.reshape(*inputs.size()[:-3], -1)
            print(inputs.shape)
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
            print(predictions)
            print(predictions.shape)
            print(f"predictions: {predictions.shape}", flush=True)
            out = predictions[1].view(-1).detach().cpu().numpy()
            print(f"out: {out.shape}")
            dl_fields["Python-DL-FIELD-OUTPUT"][curr_pos:curr_pos+vectorLen] = out
            curr_pos = curr_pos + vectorLen
        print(f"dlfields: {len(dl_fields["Python-DL-FIELD-OUTPUT"])}", flush=True)
        
        dll.send(dl_fields)  
        count = 0
        flat = []

    dll.finalize()

if __name__ == "__main__":
    main()