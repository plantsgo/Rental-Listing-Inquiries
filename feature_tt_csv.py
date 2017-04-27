#encoding=utf8
import pandas as pd
import math

train=pd.read_csv("../train.csv")
test=pd.read_csv("../test.csv")
test["interest_level"]="nnnn"
df=train.append(test)
df=df.fillna("0")

df["photo_num"]=map(lambda x:len(x.split(",")) if x!="[]" else 0,df["photos"])
df["feature_num"]=map(lambda x:len(x.split(",")) if x!="[]" else 0,df["features"])

a=40.705628
b=-74.010278
#distance to the doc select
df["distance"]=map(lambda x,y:int(math.sqrt((x-b)**2+(y-a)**2)*111),df["longitude"],df["latitude"])

df["num_description_words"] = df["description"].apply(lambda x: len(str(x).split(" ")))

df["created"] = pd.to_datetime(df["created"])
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
df["created_hour"] = df["created"].dt.hour

def time_long(x,y):
    if x==4:
        return y
    if x==5:
        return 30+y
    if x==6:
        return 30+31+y
df["time"]=map(lambda x,y:time_long(x,y),df["created_month"],df["created_day"])

df["price_bed"] = df["price"]/(df["bedrooms"]+1)
df["price_bath"] = df["price"]/(df["bathrooms"]+1)
df["price_bath_bed"] = df["price"]/(df["bathrooms"] + df["bedrooms"]+1)
df["bed_bath_dif"] = df["bedrooms"]-df["bathrooms"]

df["bed_bath_per"] = df["bedrooms"]/df["bathrooms"]
df["room_sum"] = df["bedrooms"]+df["bathrooms"]
df["bed_all_per"] = df["bedrooms"]/df["room_sum"]

#供求关系(这里可能需要组合一下构造，比如某街道同时满足房间数，某月某街道同时满足房间数，类似于求平均房价那里)
#counts of these
display=df["display_address"].value_counts()
manager_id=df["manager_id"].value_counts()
building_id=df["building_id"].value_counts()
street=df["street_address"].value_counts()
bedrooms=df["bedrooms"].value_counts()
bathrooms=df["bathrooms"].value_counts()
days=df["time"].value_counts()

df["display_count"]=map(lambda x:display[x],df["display_address"])
df["manager_count"]=map(lambda x:manager_id[x],df["manager_id"])    #经理所发的数目
df["building_count"]=map(lambda x:building_id[x],df["building_id"])
df["street_count"]=map(lambda x:street[x],df["street_address"])
df["bedrooms_count"]=map(lambda x:bedrooms[x],df["bedrooms"])
df["bathrooms_count"]=map(lambda x:bathrooms[x],df["bathrooms"])
df["day_count"]=map(lambda x:days[x],df["time"])

#经理的活跃程度
#how many days the manager active
add=pd.DataFrame(df.groupby(["manager_id"]).time.nunique()).reset_index()
add.columns=["manager_id","manager_active"]
df=df.merge(add,on=["manager_id"],how="left")

#经理拥有多少个不同的房子
#how many buildings the manager own
add=pd.DataFrame(df.groupby(["manager_id"]).building_id.nunique()).reset_index()
add.columns=["manager_id","manager_building"]
df=df.merge(add,on=["manager_id"],how="left")

#房子数比所有发出
df["manager_building_post_rt"]=df["manager_building"]/df["manager_count"]

#平均每天处理多少房子
df["build_day"]=df["manager_building"]/df["manager_active"]

#经理的活动范围
#the range place manager active
managet_place={}
for man in list(manager_id.index):
    la=df[df["manager_id"] == man]["latitude"].copy()
    lo=df[df["manager_id"] == man]["longitude"].copy()
    managet_place[man]=10000*((la.max()-la.min())*(lo.max()-lo.min()))
df["manager_place"]=map(lambda x:managet_place[x],df["manager_id"])

df["midu"]=df["manager_building"]/df["manager_place"]

#相同的房子被多少不同经理拥有（效果不大，提升很小）
#the building own by how many manager
add=pd.DataFrame(df.groupby(["building_id"]).manager_id.nunique()).reset_index()
add.columns=["building_id","building_manager"]
df=df.merge(add,on=["building_id"],how="left")

#经理当天发了多少个信息(效果不大，提升很小)
#the manager post how many listings that day
add=pd.DataFrame(df.groupby(["time","manager_id"]).listing_id.count()).reset_index()
add.columns=["time","manager_id","day_manager"]
df=df.merge(add,on=["time","manager_id"],how="left")
#每个经理当天的listing数比上今天所有的listing总数
df["day_manager_rt"]=df["day_manager"]/df["day_count"]

#每个building对应相同型号房子的均价(稍后用来做经理的开价程度)
#the building have same bedrooms and bathrooms,
add=pd.DataFrame(df.groupby(["building_id","bedrooms","bathrooms"]).price.median()).reset_index()
add.columns=["building_id","bedrooms","bathrooms","building_mean"]
df=df.merge(add,on=["building_id","bedrooms","bathrooms"],how="left")

#每个经纬度分类一下（1平方千米一个类）
df["jwd_class"]=map(lambda x,y:(int(x*100)%100)*100+(int(-y*100)%100),df["latitude"],df["longitude"])

df=df.drop(["photos","id","created"],axis=1)
df.to_csv("hello.csv",index=None)
