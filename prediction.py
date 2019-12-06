import os
import xlrd
import numpy as np
import pandas as pd
import sklearn.preprocessing
from config import *

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False):
    import cnn
    return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)

wb = xlrd.open_workbook(loc)

sheet_names = wb.sheet_names()

# number of models
nModels = 19

result_sheets = []
result_vals = []
acc_vals = []
for sheetIdx in range(0, len(sheet_names), 2):
    sheet = wb.sheet_by_index(sheetIdx)
    print("=============================================================")
    print(sheet.name)

    tempx = []
    tempy = []
    start_rowIdx = 3
    for rowIdx in reversed(range(start_rowIdx, sheet.nrows - 8)):
        actual_result = sheet.cell_value(rowIdx, 1)
        if( actual_result != 0 and actual_result != 1 ):
            continue

        x_row = []
        if (use_V):
            val = sheet.cell_value(rowIdx, 21)
            if val == "" or val == "NA":
                val = 0.5
            x_row.append(val)

        if (use_W):
            val = sheet.cell_value(rowIdx, 22)
            if val == "" or val == "NA":
                val = 0.5
            x_row.append(val)

        if (use_X):
            val = sheet.cell_value(rowIdx, 23)
            if val == "" or val == "NA":
                val = 0.5
            x_row.append(val)

        if (use_Y):
            val = sheet.cell_value(rowIdx, 24)
            if val == "" or val == "NA":
                val = 0.5
            x_row.append(val)

        for mIdx in range(nModels):
            val = sheet.cell_value(rowIdx, 32 + 6 * mIdx)
            if val == "" or val == "NA":
                val = 0.5
            x_row.append(val)

            for dIdx in range(nColModel):
                val = sheet.cell_value(rowIdx, 33 + 6 * mIdx + dIdx)
                vals = val[1:len(val)-1].split("/")
                if vals[1] == "0":
                    x_row.append(0)
                else:
                    x_row.append(int(vals[0])/int(vals[1]))

        tempx.append(x_row)
        tempy.append(actual_result)

    x_row = []
    for mIdx in range(nModels):
        val = sheet.cell_value(start_rowIdx - 1, 32 + 6 * mIdx)
        if val == "" or val == "NA":
            val = 0.5
        x_row.append(val)

        for dIdx in range(nColModel):
            val = sheet.cell_value(start_rowIdx - 1, 33 + 6 * mIdx + dIdx)
            vals = val[1:len(val)-1].split("/")
            if vals[1] == "0":
                x_row.append(0)
            else:
                x_row.append(int(vals[0])/int(vals[1]))

    tempx.append(x_row)

    prediction_result = 0


    y = tempy[nHistCount:]
    x = []
    for i in range(nHistCount, len(tempx)):
        tempi = tempx[i - nHistCount:i]
        tempi = np.array(tempi).T.tolist()
        x.append(tempi)

    x = np.asarray(x)
    y = np.asarray(y)

    totalLen = y.shape[0]
    #splitted = int(totalLen * 0.1)
    splitted = totalLen - 1

    x_train = x[:splitted]
    x_test = x[splitted:splitted + 1]
    x_input = x[splitted + 1:]
    y_train = y[:splitted]
    y_test = y[splitted:]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()


    input_shape = x_train.shape[1:]

    classifier_name = "cnn"
    output_directory = "result/" + sheet.name + "/"
    create_directory(output_directory)
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    prediction_result, acc = classifier.fit(x_train, y_train, x_test, y_test, x_input, batch_size, nEpochs)

    if( prediction_result > 0.5 ):
        prediction_result = 1
    else:
        prediction_result = 0

    result_sheets.append(sheet.name)
    result_vals.append(prediction_result)
    acc_vals.append(acc)

    print ( "result : " + str(prediction_result))
    print("=============================================================")
    print()

df = pd.DataFrame(data={"security": result_sheets, "prediction": result_vals, "accuracy": acc_vals})
df.to_csv("result/result.csv", sep=',',index=False)