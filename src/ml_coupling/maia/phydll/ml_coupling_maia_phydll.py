import mpi4py
mpi4py.rc.thread_level = "funneled"

import mpi4py.MPI as MPI

from pyphydll.pyphydll import PhyDLL
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
import scorep


import thoplw

current_dir = os.path.dirname(os.path.abspath(__file__))
python_utils_path = os.path.abspath(os.path.join(current_dir, '../../../python_utils/'))
if python_utils_path not in sys.path:
    sys.path.insert(0, python_utils_path)

from TimerManager import TimerManager

tm = TimerManager()

def main():
    tm.start("Init")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    print(f"Torch Threads: {torch.get_num_threads()}", flush=True)

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
        num_cells_per_process.append(meta_info_field[pid][2] // sequence_len // field_count) #sequenceLen * nFields * numCubes * cubeSize
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
    model.eval()

    ##################
    # Inference
    ##################
    curr_seq_idx = 0

    phy_fields = [[[] for _ in range(num_phy_procs)] for _ in range(field_count)]
    total_cells = sum(num_cells_per_process)
    
    dl_fields = [
        np.zeros(total_cells),
        np.zeros(total_cells),
        np.zeros(total_cells),
    ]
    tm.stop("Init")
    tm.start("Phydll")
    
    while dll.is_phy_signal():
        tm.start("recv")
        curr_seq_idx+=1
        
        fields = dll.recv()
        
        split_points = np.cumsum(num_cells_per_process)[:-1]
        
        # pid
        proc_chunks0 = np.split(fields["Python-DL-FIELD-INPUT-0"], split_points)
        proc_chunks1 = np.split(fields["Python-DL-FIELD-INPUT-1"], split_points)
        proc_chunks2 = np.split(fields["Python-DL-FIELD-INPUT-2"], split_points)
        
        for pid in range(num_phy_procs):
            phy_fields[0][pid].append(proc_chunks0[pid])
            phy_fields[1][pid].append(proc_chunks1[pid])
            phy_fields[2][pid].append(proc_chunks2[pid])
            # [fields][pid][seq][numcubes]

        tm.stop("recv")
        #Skip the next stuff if not yet gotten all sequences
        if curr_seq_idx < sequence_len:
            continue 

            
        tm.start("Main")
        #Reshape each chunk into [sequenceLen, vectorLen]
        curr_pos = 0 #Increases by num_cells_per_process[pid] // seqlen
        cubeSize = cubeD**3
        #for pid, flat in enumerate(per_proc_flat_data):
        for pid in range(num_phy_procs):
            tm.start("process_data")
            vectorLen = num_cells_per_process[pid]
            
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
            tm.stop("process_data")
            # Now flatten cubes and features per timestep to get shape (seq_len, num_cubes * features)
            #fields_data = fields_data.reshape(seq_len, -1)

            # Convert to tensor
            tm.start("inference")
            inputs = torch.tensor(fields_data, dtype=torch.float32, device=device)
            inputs = inputs.reshape(*inputs.size()[:-3], -1) #seqlen, fields*numcubes, cubeD^3
            
            with torch.no_grad():
                tm.start("run_inf")
                predictions = run_encoder_decoder_inference(
                    device=device,
                    model=model,
                    src=inputs,
                    forecast_window=forecast_window,
                    batch_size=inputs.shape[1],
                    batch_first=False
                )
                tm.stop("run_inf")
                
                out = predictions[1].view(-1).detach().cpu().numpy()
                out_reshaped = out.reshape(fields, num_cubes, cubeD, cubeD, cubeD)

                # Now flatten cube dims per field back to vector length
                # Each field data will be shape: (num_cubes * cubeD^3) == vectorLen
                for field in range(fields):
                    field_data = out_reshaped[field].reshape(-1)
                    dl_fields[field][curr_pos:curr_pos+vectorLen] = field_data
            curr_pos = curr_pos + vectorLen    
            tm.stop("inference")
        tm.stop("Main")

        tm.start("send")
        dl_fields_send = {
            "Python-DL-FIELD-OUTPUT-0": dl_fields[0],
            "Python-DL-FIELD-OUTPUT-1": dl_fields[1],
            "Python-DL-FIELD-OUTPUT-2": dl_fields[2]
        }
        dll.send(dl_fields_send)

        #Reset data
        for field in range(field_count):
            for pid in range(num_phy_procs):
                phy_fields[field][pid].clear()
        curr_seq_idx = 0 
        tm.stop("send")

    dll.finalize()

    tm.stop("Phydll")
    tm.summary()

if __name__ == "__main__":
    main()
    MPI.Finalize()