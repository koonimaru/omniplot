import glob

fs=glob.glob("/home/koh/public/konimaru/Archive_data/IDR/Figures/*/*/*txt")
id_file_dict={}
produced={}
for f in fs:
    with open(f) as fin:
        for li, l in enumerate(fin):
            l=l.split()
            for i in range(len(l)):
                if l[i].startswith("#INPUT"):
                    id_file_dict[l[i].strip(",")]=l[i-1]
                elif l[i-1]=="Producing" and l[i].startswith("INPUT"):
                    produced[l[i]]=f+", line number: "+str(li+1)
with open("/home/koh/public/konimaru/Archive_data/IDR/data_id.txt", "w") as fout:
    keys=sorted(id_file_dict.keys())
    for key in keys:
        fout.write("\t".join([key,id_file_dict[key], produced[key.strip("#")]+"\n"]))