import csv

first_list = []
with open('total_fmap_means_comp.csv', newline='') as firstfile:
    csvreader = csv.reader(firstfile, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if (i > 0):
            first_list.append(row[0])
        i += 1

second_list = []
with open('total_fmap_means_nocomp.csv', newline='') as secondfile:
    csvreader = csv.reader(secondfile, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if (i > 0):
            second_list.append(row[0])
        i += 1

union_list = []
union_list_length = 128

j = 0
while len(union_list) < union_list_length:
    if (j % 2 == 0):
        item = first_list.pop(0)
    else:
        item = second_list.pop(0)
    if item not in union_list:
        union_list.append(item)
    j += 1

with open('union_means_list.csv', 'w', newline='') as csvfile:
    resultcsv = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key in union_list:
        resultcsv.writerow([key])