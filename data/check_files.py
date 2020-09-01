import os.path
import csv
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        for i in range(3):
           source_path = line[i]
           file_name = "./" + source_path.split('\\')[-1]
           if not os.path.isfile(file_name):
               print("%s does not exist !!!" % file_name)