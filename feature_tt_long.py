#encoding=utf8
from operator import itemgetter
import numpy as np
import pandas as pd
import operator
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import math
def get_data():
    all=pd.read_csv("hello.csv").fillna(-1)
    #经纬度附近的房子中价格比这个低的个数（需要优化）
    all["jwd_type_low_than_num"]=map(lambda lo,la,ba,be,p:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)&(all.price<=p)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"],all["price"])
    all["jwd_type_all"]=map(lambda lo,la,ba,be:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"])
    all["jwd_type_rt"]=all["jwd_type_low_than_num"]/all["jwd_type_all"]

    #所有building_id=0的经纬度坐标：
    building_zeros_la=list(all[all.building_id.astype("str")=="0"].latitude)
    building_zeros_lo=list(all[all.building_id.astype("str")=="0"].longitude)
    building_zeros=zip(building_zeros_la,building_zeros_lo)
    def building_zero_num(la,lo,n):
        num=0
        for s in building_zeros:
            slo=float(s[1])
            sla=float(s[0])
            dis=math.sqrt((la-sla)**2+(lo-slo)**2)*111
            if dis<=n:
                num+=1
        return num

    # 半径一公里内有多少building_id 为0的
    print "半径一公里内有多少building_id 为0的"
    #需要优化
    all["building_zero_num"] = map(lambda la, lo: building_zero_num(la, lo,1), all["latitude"], all["longitude"])

    all=all[["jwd_type_low_than_num","jwd_type_all","jwd_type_rt","building_zero_num","listing_id"]]

    all.to_csv("timeout.csv",index=None)

get_data()