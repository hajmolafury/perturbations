import csv
import globalV

def write_in_file(failed, dir, onlyEpochs):
    params = ["depth", str(globalV.n_hl), "width", str(globalV.hl_size), "learning rate", str(globalV.learning_rate),
              "n_epochs", str(globalV.n_epochs), "alpha", str(globalV.alpha)]

    with open(dir, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)

        row = globalV.epochs977
        if(row == None):
            row = ["Failed"]
        writer.writerow(row)
        row = globalV.epochs979
        if (failed):
            row = ["Failed"]
        writer.writerow(row)
        if (not failed and not onlyEpochs):
            row = globalV.test_acc
            writer.writerow(row)

        csvFile.flush()

    csvFile.close()
    return
