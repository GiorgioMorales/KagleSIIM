import csv
import cv2
from mask_functions import *

maskname = []
maskarray = []

with open('train-rle.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    print("leyendo")
    for row in csv_reader:
        if line_count != 0:

            line_count += 1
            print(line_count)

            maskname.append(row[0])
            maskarray.append(row[1])

            if not(row[1] == ' -1' or row[1] == '-1'):
                mask = rle2mask(row[1], 1024, 1024)
                cv2.imwrite('C://Users//User//KagleSIIM//MasksO//' + str(row[0]) + ".tif", mask)

        else:
            line_count += 1

masknamea = np.array(maskname)
masknameunique = np.unique(masknamea)
print(len(masknameunique))