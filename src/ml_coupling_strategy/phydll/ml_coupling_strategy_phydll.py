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

    field_count = 3 # number of DL fields
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
        num_cells_per_process.append(meta_info_field[pid][2] // sequence_len // field_count)
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
    #model = torch.compile(model)
    model.eval()

    ##################
    # Inference
    ##################
    curr_seq_idx = 0
    phy_fields = [
        [[] * num_phy_procs],
        [[] * num_phy_procs],
        [[] * num_phy_procs],
    ]
    total_cells = sum(num_cells_per_process)
    print(total_cells)
    dl_fields = [
        np.zeros(total_cells),
        np.zeros(total_cells),
        np.zeros(total_cells),
    ]
    print(dl_fields)
    while dll.is_phy_signal():
        curr_seq_idx+=1
        
        fields = dll.recv()
        
        split_points = np.cumsum(num_cells_per_process)[:-1]
        print(num_cells_per_process)
        print(split_points)
        # pid
        perproc0 = np.split(fields["Python-DL-FIELD-INPUT-0"], split_points)[0]
        perproc1 = np.split(fields["Python-DL-FIELD-INPUT-1"], split_points)[0]
        perproc2 = np.split(fields["Python-DL-FIELD-INPUT-2"], split_points)[0]
        print(perproc0)
        for pid in range(num_phy_procs):
            phy_fields[0][pid].append(perproc0)
            phy_fields[1][pid].append(perproc1)
            phy_fields[2][pid].append(perproc2)
            # [fields][pid][seq][numcubes]

        #Skip the next stuff if not yet gotten all sequences
        if curr_seq_idx < sequence_len:
            continue 

        #Split data per process
        """split_points = np.cumsum(num_cells_per_process)[:-1]
        per_proc_flat_data = np.split(phy_fields, split_points)"""

        #Reshape each chunk into [sequenceLen, vectorLen]
        curr_pos = 0 #Increases by num_cells_per_process[pid] // seqlen
        nFields = 3
        cubeSize = cubeD**3
        #for pid, flat in enumerate(per_proc_flat_data):
        for pid in range(num_phy_procs):
            vectorLen = num_cells_per_process[pid]
            """reshaped = flat.reshape(sequence_len, vectorLen)

            inputs = torch.tensor(reshaped, dtype=torch.float32, device=device)
            inputs = inputs.reshape(sequence_len, int(vectorLen / (cubeD**3)), cubeD, cubeD, cubeD)
            inputs = inputs.reshape(*inputs.size()[:-3], -1)

            input_cubes = inputs.permute(1, 0, 2)  # shape: [num_cubes, sequence_len, cube_features]
            input_cubes_flat = input_cubes.reshape(input_cubes.shape[0], -1).cpu().numpy()"""

            """arr = np.asarray(flat, dtype=np.float32).reshape(sequence_len, vectorLen)

            # total_per_timestep = num_cubes * nFields * cubeSize
            total_per_timestep = arr.shape[1]
            # deduce how many cubes for this process:
            num_cubes = total_per_timestep // (nFields * cubeSize)

            # We'll build an array [seq_len, num_cubes, nFields, cubeSize]
            cubes_4d = np.zeros((sequence_len, num_cubes, nFields, cubeSize), dtype=np.float32)

            # Demultiplex: each time step t, each cube i, each field f
            for t in range(sequence_len):
                row_data = arr[t]  # shape [num_cubes * nFields * cubeSize]
                for i in range(num_cubes):
                    base_cube = i * nFields * cubeSize
                    for f in range(nFields):
                        offset = base_cube + f*cubeSize
                        cubes_4d[t, i, f, :] = row_data[offset : offset + cubeSize]

            # Reshape the last dim to (cubeD,cubeD,cubeD) => [seq_len, num_cubes, nFields, cubeD, cubeD, cubeD]
            cubes_5d = cubes_4d.reshape(sequence_len, num_cubes, nFields, cubeD, cubeD, cubeD)
            #[5, num_cubes, 3, a, a, a]
            # If your model wants shape [sequence_len, batch_size, nFields*cubeSize], do:
            # => [5, num_cubes, 3*cubeD^3]
            inputs = torch.tensor(cubes_5d, dtype=torch.float32, device=device)
            #inputs = inputs.view(sequence_len, num_cubes, nFields*cubeSize)
            print(inputs.shape)#torch.Size([5, 690, 1536])
            inputs = inputs.reshape(sequence_len,num_cubes*nFields,cubeD,cubeD,cubeD)
            inputs = inputs.reshape(*inputs.size()[:-3], -1)#.to(device)
            print(inputs.shape)"""
            # Check for duplicate input cubes (optional). 
            # We'll flatten each cube to 1D, ignoring time dimension => shape [num_cubes, (sequence_len * nFields*cubeSize)]
            """cubes_for_dup = inputs.permute(1,0,2).reshape(num_cubes, -1).cpu().numpy()
            duplicate_input_indices = []
            for i in range(num_cubes):
                for j in range(i + 1, num_cubes):
                    if np.allclose(cubes_for_dup[i], cubes_for_dup[j]):
                        duplicate_input_indices.append((i, j))

            if duplicate_input_indices:
                print(f"[Rank {lrank}] (PID={pid}) Found duplicate input cubes at indices: {duplicate_input_indices}")"""
            #cubes_per_process[pid][seq_idx][fields]
            """for field in range(field_count):
                print(phy_fields[field][pid],flush=True)
                data = np.array(phy_fields[field][pid])
                print(data.shape, flush=True)
                data = data.reshape(sequence_len, (vectorLen//cubeSize), cubeD, cubeD, cubeD)
                print(data.shape, flush=True)
                inputs = torch.tensor(data, dtype=torch.float32, device=device)
                print(inputs.shape, flush=True)
                inputs = inputs.reshape(*inputs.size()[:-3], -1)
                print(inputs.shape, flush=True)
                # Run model inference
                #Input is [sequence_len, num_cubes * cubeD * cubeD * cubeD]"""
            
            fields_data = []
            for field in range(field_count):
                data = np.array(phy_fields[field][pid])
                data = data.reshape(sequence_len, (num_cells_per_process[pid]//cubeSize), cubeD, cubeD, cubeD)
                fields_data.append(data)
            # Stack along new 0th axis for fields
            fields_data = np.stack(fields_data, axis=0)  # shape: [fields, seq_len, num_cubes, cubeD, cubeD, cubeD]
            
            # Permute axes so fields is after seq_len: (seq_len, fields, num_cubes, cubeD, cubeD, cubeD)
            fields_data = np.transpose(fields_data, (1, 0, 2, 3, 4, 5))
            
            # Flatten cube dims + fields dimension into feature dim: 
            # new shape: (seq_len, num_cubes, fields * cubeD * cubeD * cubeD)
            seq_len, fields, num_cubes, d1, d2, d3 = fields_data.shape
            fields_data = fields_data.reshape(seq_len, fields * num_cubes, d1, d2, d3)
            
            # Now flatten cubes and features per timestep to get shape (seq_len, num_cubes * features)
            #fields_data = fields_data.reshape(seq_len, -1)
            print(fields_data.shape)
            # Convert to tensor
            inputs = torch.tensor(fields_data, dtype=torch.float32, device=device)
            inputs = inputs.reshape(*inputs.size()[:-3], -1) #seqlen, fields*numcubes, cubeD^3
            with torch.no_grad():
                predictions = run_encoder_decoder_inference(
                    device=device,
                    model=model,
                    src=inputs,
                    forecast_window=forecast_window,
                    batch_size=inputs.shape[1],
                    batch_first=False
                )
                print(predictions.shape, flush=True)
                out = predictions[1].view(-1).detach().cpu().numpy()
                out_reshaped = out.reshape(fields, num_cubes, cubeD, cubeD, cubeD)

                # Now flatten cube dims per field back to vector length
                # Each field data will be shape: (num_cubes * cubeD^3) == vectorLen
                for field in range(fields):
                    field_data = out_reshaped[field].reshape(-1)
                    dl_fields[field][curr_pos:curr_pos+vectorLen] = field_data
                    #print(dl_fields[field], flush=True)
                    #print(dl_fields[field].shape)
                    #print(curr_pos)
                    #print(vectorLen, flush=True)
                    #dl_fields[field][curr_pos:curr_pos+vectorLen] = out
            curr_pos = curr_pos + vectorLen    

            """pred_cubes = predictions[1].permute(1, 0, 2)  # shape: [num_cubes, forecast_window, features]
                pred_cubes_flat = pred_cubes.reshape(pred_cubes.shape[0], -1).cpu().numpy()

                duplicate_pred_indices = []
                for i in range(len(pred_cubes_flat)):
                    for j in range(i + 1, len(pred_cubes_flat)):
                        if np.allclose(pred_cubes_flat[i], pred_cubes_flat[j]):
                            duplicate_pred_indices.append((i, j))

                if duplicate_pred_indices:
                    print(f"[Rank {lrank}] Found duplicate predicted cubes at indices: {duplicate_pred_indices}")"""
        dl_fields_send = {
            "Python-DL-FIELD-OUTPUT-0": dl_fields[0],
            "Python-DL-FIELD-OUTPUT-1": dl_fields[1],
            "Python-DL-FIELD-OUTPUT-2": dl_fields[2]
        }
        dll.send(dl_fields_send)

        #Reset data
        phy_fields = [
            [[] * num_phy_procs],
            [[] * num_phy_procs],
            [[] * num_phy_procs],
        ]
        total_cells = sum(num_cells_per_process)
        dl_fields = [
            np.zeros(total_cells),
            np.zeros(total_cells),
            np.zeros(total_cells),
        ]
        curr_seq_idx = 0

    dll.finalize()

if __name__ == "__main__":
    main()