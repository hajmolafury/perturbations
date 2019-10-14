import csv
import globalV

def write_in_file(failed, dir, onlyEpochs):
    params = ["depth", str(globalV.n_hl), "width", str(globalV.hl_size), "learning rate", str(globalV.learning_rate),
              "n_epochs", str(globalV.n_epochs), "alpha", str(globalV.alpha)]

    with open(dir, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)

        if(onlyEpochs):
            row = globalV.epochs977
        else:
            row = globalV.test_acc

        if (failed):
            row = ["Failed"]

        writer.writerow(row)
        if (onlyEpochs):
            row = globalV.epochs979
            writer.writerow(row)

        csvFile.flush()

    csvFile.close()
    return
