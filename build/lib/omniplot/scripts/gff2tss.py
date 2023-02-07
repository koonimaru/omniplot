import sys
import os 



def writetss(f):
    foname=os.path.splitext(f)[0]+"_tss.bed"
    
    tmp=[]
    with open(f) as fin, open(foname,"w") as fout:
        for l in fin:
            if l.startswith("#"):
                continue
            l=l.split()
            if l[2]=="transcript":
                chrom, s, e, name, ori=l[0],str(int(l[3])-1),l[4], l[8], l[6]
                tmp.append([chrom, int(s), int(e), name, ori])
        tmp = sorted(tmp, key = lambda x: (x[1], x[2]))
        for chrom, s, e, name, ori in tmp:
            fout.write("\t".join([chrom, str(s), str(e), name, ori+"\n"]))

def main(f):
    writetss(f)

if __name__=="__main__":
    f=sys.argv[1]
    main(f)