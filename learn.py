import csv
with open("data/sss.csv") as file:
    row = csv.reader(file)
    row = list(row)
    
    # print(row[0][-1])
    readed_list=[]
    readed_list.append(0)
    for n in row[0]:
        if n == 'ï»¿0':
            n = n[3:]
        
        try:
            readed_list.append(int(n))
        except:
            readed_list.append(float(n))
    print(readed_list)