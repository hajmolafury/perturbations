import csv
import globalV

def write_in_file(failed, dir):
    row = globalV.test_acc
    # print(test_acc)
    params = ["depth", str(globalV.n_hl), "width", str(globalV.hl_size), "learning rate", str(globalV.learning_rate), "n_epochs",
              str(globalV.n_epochs), "alpha", str(globalV.alpha)]

    with open(dir, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        if (failed):
            row = ["Failed"]
        writer.writerow(row)
        csvFile.flush()

    csvFile.close()
    return
