import pandas as pd
import numpy as np
import os
import csv
def overlap(x1,w1,x2,w2):
	l1 = x1 - w1/2
	l2 = x2 - w2/2
	left = np.maximum(l1 , l2)
	r1 = x1 + w1/2
	r2 = x2 + w2/2
	right=np.minimum(r1 , r2)
	return right-left

def box_intersection(x1,w1,y1,h1,x2,w2,y2,h2):
	w=overlap(x1,w1,x2,w2)
	h=overlap(y1,h1,y2,h2)
	w=np.maximum(w,0)
	h=np.maximum(h,0)
	return w*h

def box_union(x1,w1,y1,h1,x2,w2,y2,h2):
	area=box_intersection(x1,w1,y1,h1,x2,w2,y2,h2)
	return w1*h1+w2*h2-area

files=["Illegal_parking","laji","outManagement"]
total=0
recognize=0
true_ng_sum=0
false_ps_sum=0
true_ps_sum=0
false_ng_sum=0

for file in files:
	# if os.path.exists(file)==False or os.path.getsize(file)==0 or file.endswith(".txt")==False:
	# 	continue
	data = pd.read_csv(file+".txt",sep=" ",header=None)
	#print(data.describe())
	# if(os.path.exists("../"+file)==False):
	# 	false_ng=false_ng+data.describe()["class"].get("count")
	# 	continue
	fact = pd.read_csv(file+"_labels.txt",sep=" ",header=None)
	data.columns=["image","class","rate","xmin","ymin","xmax","ymax"]
	fact.columns=["image","xmin2","ymin2","xmax2","ymax2","class"]
	fact=fact.drop(["xmin2", "ymin2","xmax2","ymax2"], axis=1)
	result=pd.merge(data, fact, on=['image','class'])
	result=result.drop_duplicates()
	# w=abs(result["xmax"]-result["xmin"])
	# w2=abs(result["xmax2"]-result["xmin2"])
	# h=abs(result["ymax"]-result["ymin"])
	# h2=abs(result["ymax2"]-result["ymin2"])
	# union_size=box_union((result["xmin"]+result["xmax"])/2,w,(result["ymin"]+result["ymax"])/2,h,(result["xmin2"]+result["xmax2"])/2,w2,(result["ymin2"]+result["ymax2"])/2,h2)
	# cross_size=box_intersection((result["xmin"]+result["xmax"])/2,w,(result["ymin"]+result["ymax"])/2,h,(result["xmin2"]+result["xmax2"])/2,w2,(result["ymin2"]+result["ymax2"])/2,h2)
	# IOU=cross_size/union_size
	# IOU_frame=pd.DataFrame({"IOU":IOU},index=[0])
	
	data_desc=data.describe()
	fact_desc=fact.describe()		
	result_desc=result.describe()
	#result_desc=result_desc.append(IOU_frame)
	recognize=recognize+data_desc["class"].get("count")
	true_ps=result_desc["class"].get("count")
	true_ps_sum=true_ps_sum+true_ps
	false_ps=data_desc["class"].get("count")-true_ps
	false_ps_sum=false_ps_sum+false_ps
	false_ng=fact_desc["class"].get("count")-true_ps
	false_ng_sum=false_ng_sum+false_ng
	total=total+pd.merge(data, fact,how="right",on=['image','class']).describe()["class"].get("count")
	true_ng=0
	accuracy=(true_ps+true_ng)/(true_ps+true_ng+false_ng+false_ps)
	precision=true_ps/(true_ps+false_ps)
	recall=true_ps/(true_ps+false_ng)
	F1=2/((1/precision)+(1/recall))

	per_frame=pd.DataFrame({"count":data_desc["class"].get("count"),"true_ps":true_ps,"true_ng":0,"false_ng":false_ng,"false_ps":false_ps,"recognize":data_desc["class"].get("count")/pd.merge(data, fact,how="right",on=['image','class']).describe()["class"].get("count"),"recall":recall,"precision":precision,"accuracy":accuracy,"F1":F1},index=[0])
	per_frame.to_html(file+".html")

accuracy=(true_ps_sum+true_ng_sum)/(true_ps_sum+true_ng_sum+false_ng_sum+false_ps_sum)
precision=true_ps_sum/(true_ps_sum+false_ps_sum)
recall=true_ps_sum/(true_ps_sum+false_ng_sum)
F1=2/((1/precision)+(1/recall))
frame=pd.DataFrame({"count":recognize,"true_ps_sum":true_ps_sum,"true_ng_sum":0,"false_ng_sum":false_ng_sum,"false_ps_sum":false_ps_sum,"recognize":recognize/total,"recall":recall,"precision":precision,"accuracy":accuracy,"F1":F1},index=[0])
frame.to_html("total.html")



