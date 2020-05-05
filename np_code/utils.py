import csv
import numpy as np
import globalV
import argparse
from datetime import datetime, timedelta

def file_writer(dir_path, elapsed_time, write_norms=False):
    params = ["update_rule",globalV.update_rule.upper(),"depth", str(globalV.n_hl), "width", str(globalV.hl_size), "lr", str(globalV.lr),
              "n_epochs", str(globalV.n_epochs), "batch_size",str(globalV.batch_size)]

    with open(dir_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        
        if(write_norms):
            for hl in range(globalV.n_hl+1):
                writer.writerow(globalV.w_norm[str(hl)])
        
        writer.writerow("")

        if(write_norms):
            for hl in range(globalV.n_hl+1):
                writer.writerow(globalV.b_sum[str(hl)])
        
        writer.writerow(globalV.test_acc)
        writer.writerow(str(elapsed_time))
        writer.writerow("")
        csvFile.flush()

    csvFile.close()
    return

def sma_accuracy(period):
    return np.ma.average(globalV.test_acc[-int(period):])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", type=float)
    ap.add_argument("-update_rule")
    ap.add_argument("-n_hl", type=int)
    ap.add_argument("-hl_size", type=int)
    ap.add_argument("-n_epochs", type=int)
    args= ap.parse_args()

    if args.lr:
        globalV.lr=args.lr
    if args.update_rule:
        globalV.update_rule=args.update_rule
    if args.n_hl:
        globalV.n_hl=args.n_hl
    if args.hl_size:
        globalV.hl_size=args.hl_size
    if args.n_epochs:
        globalV.n_epochs=args.n_epochs

def get_elapsed_time(sec):
    sec=timedelta(seconds=int(sec))
    d = datetime(1,1,1) + sec
    if(d.hour>0):
        return str(d.hour)+"hours"+ str(d.minute)+"min "+ str(d.second)+"sec"
    else:
        return str(d.minute)+"min "+ str(d.second)+"sec"
    
