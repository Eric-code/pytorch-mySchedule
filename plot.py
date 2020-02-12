import matplotlib.pyplot as plt


y1 = []
y2 = []
y3 = []

with open("poktr_20_origin_2.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        all_datas = line.split("|")
        datas = all_datas[3]
        data = int(datas.split(":")[1])
        y1.append(data)
        if i > 780:
            break

with open("poktr_rtmdp_20_2.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        all_datas = line.split("|")
        datas = all_datas[3]
        data = int(datas.split(":")[1])
        y2.append(data)
        if i > 780:
            break

with open("poktr_rtmdp_att_20_2.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        all_datas = line.split("|")
        datas = all_datas[3]
        data = int(datas.split(":")[1])
        y3.append(data)
        if i > 780:
            break

x = [j for j in range(782)]
plt.plot(x, y1, label="origin")
plt.plot(x, y2, label="rtmdp")
plt.plot(x, y3, label="atten")
plt.legend()
plt.show()


