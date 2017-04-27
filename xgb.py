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
    a = 40.705628
    b = -74.010278
    all=pd.read_csv("hello.csv").fillna(-1)
    #####增加listing_id与时间的斜率
    Min_lis_id=all["listing_id"].min()
    Min_time=all["time"].min()
    all["gradient"]=((all["listing_id"])-Min_lis_id)/(all["time"]-Min_time)
    ###############################
    #经理租房的开价程度
    print "开价程度"
    all["building_dif"]=all["price"]-all["building_mean"]
    all["building_rt"]=all["price"]/all["building_mean"]

    #每个经理building_rt的均值
    add = pd.DataFrame(all.groupby(["manager_id"]).building_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay"]
    all = all.merge(add, on=["manager_id"], how="left")

    #根据经纬度类别构造特征

    #区域内有多少不同经理竞争，即为经理数
    print "区域内有多少不同经理竞争"
    add = pd.DataFrame(all.groupby(["jwd_class"]).manager_id.nunique()).reset_index()
    add.columns = ["jwd_class", "manager_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    #一个经理经营多少个区域
    print "一个经理经营多少个区域"
    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_class.nunique()).reset_index()
    add.columns = ["manager_id", "manager_jwd_class"]
    all = all.merge(add, on=["manager_id"], how="left")

    #该区域内的均价
    print "区域内的均价"
    add = pd.DataFrame(all.groupby(["jwd_class"]).price.median()).reset_index()
    add.columns = ["jwd_class", "price_mean_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    #该区域内的building数
    print "该区域内的building数"
    add = pd.DataFrame(all.groupby(["jwd_class"]).building_id.nunique()).reset_index()
    add.columns = ["jwd_class", "building_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    #每个manager的平均放照片多少，描述字的多少，反映经理工作的仔细程度
    print "每个manager的平均放照片多少"
    add = pd.DataFrame(all.groupby(["manager_id"]).photo_num.mean()).reset_index()
    add.columns = ["manager_id", "manager_photo"]
    all = all.merge(add, on=["manager_id"], how="left")

    print "每个manager的平均描述字多少"
    add = pd.DataFrame(all.groupby(["manager_id"]).num_description_words.mean()).reset_index()
    add.columns = ["manager_id", "manager_desc"]
    all = all.merge(add, on=["manager_id"], how="left")

    print "每个manager的平均描述feature个数"
    add = pd.DataFrame(all.groupby(["manager_id"]).feature_num.mean()).reset_index()
    add.columns = ["manager_id", "manager_feature"]
    all = all.merge(add, on=["manager_id"], how="left")

    #各种房型的均价，然后减去当前房子的均价就是地理位置带来的影响
    add = pd.DataFrame(all.groupby(["bathrooms","bedrooms"]).price.median()).reset_index()
    add.columns = ["bathrooms","bedrooms", "fangxing_mean"]
    all = all.merge(add, on=["bathrooms","bedrooms"], how="left")

    all["fangxing_mean_dif_building"]=all["fangxing_mean"]-all["building_mean"]
    #all["fangxing_mean_rt_building"] = all["fangxing_mean"]/all["building_mean"]

    #总的平均房价,反应了由地理位置带来的房价影响(为了复原结果,貌似没什么用)
    price_mean_all=all.price.median()
    all["price_all_dif_jwd"]=price_mean_all-all["price_mean_jwd"]

    # 在某经纬度范围内各种房型的均值
    add = pd.DataFrame(all.groupby(["jwd_class","bathrooms","bedrooms"]).price.median()).reset_index()
    add.columns = ["jwd_class","bathrooms","bedrooms", "type_jwd_price_mean"]
    all = all.merge(add, on=["jwd_class","bathrooms","bedrooms"], how="left")

    #和出价比较，反应经理的房子出价和此处相配不
    all["type_jwd_price_mean_dif"]=all["price"]-all["type_jwd_price_mean"]
    all["type_jwd_price_mean_rt"]=all["price"]/all["type_jwd_price_mean"]
    #相同经纬度和不同building比较，反应feature的影响
    all["type_jwd_building_mean_dif"]=all["building_mean"]-all["type_jwd_price_mean"]
    all["type_jwd_building_mean_rt"]=all["building_mean"]/all["type_jwd_price_mean"]
    #该经纬度附近的和全市的比较，侧面反应该地区的经济发展和贵不贵
    all["fangxing_mean_dif_jwd"] = all["fangxing_mean"] - all["type_jwd_price_mean"]
    all["fangxing_mean_rt_jwd"] = all["fangxing_mean"]/all["type_jwd_price_mean"]

    #该manager在该地区出价的比的平均
    add = pd.DataFrame(all.groupby(["manager_id"]).type_jwd_price_mean_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay_jwd"]
    all = all.merge(add, on=["manager_id"], how="left")

    #该building在该地区出价的比的平均
    add = pd.DataFrame(all.groupby(["building_id"]).type_jwd_building_mean_rt.mean()).reset_index()
    add.columns = ["building_id", "building_pay_jwd"]
    all = all.merge(add, on=["building_id"], how="left")

    #该jwd在该市出价的比的平均
    add = pd.DataFrame(all.groupby(["jwd_class"]).fangxing_mean_rt_jwd.mean()).reset_index()
    add.columns = ["jwd_class", "jwd_pay_all"]
    all = all.merge(add, on=["jwd_class"], how="left")

    #该经理拥有的房子比周边贵还是便宜
    add = pd.DataFrame(all.groupby(["manager_id"]).building_pay_jwd.mean()).reset_index()
    add.columns = ["manager_id", "manager_own_ud"]
    all = all.merge(add, on=["manager_id"], how="left")

    #该经理拥有的房子地区比该市贵还是便宜
    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_pay_all.mean()).reset_index()
    add.columns = ["manager_id", "manager_own_ud_all"]
    all = all.merge(add, on=["manager_id"], how="left")

    #该经理拥有的房子比该市贵还是便宜
    all["manager_building_all_rt"]=all["manager_own_ud"]/all["manager_own_ud_all"]

    #原来是在这里聚类的。。。因为顺序错了所以未复现
    #再聚类看看？
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=5,random_state=1)
    #去除异常点
    all["longitude"]=map(lambda x:-73.75 if x>=-73.75 else x,all["longitude"])
    all["longitude"]=map(lambda x:-74.05 if x<=-74.05 else x,all["longitude"])
    all["latitude"]=map(lambda x:40.4 if x<=40.4 else x,all["latitude"])
    all["latitude"]=map(lambda x:40.9 if x>=40.9 else x,all["latitude"])
    data=all[["latitude","longitude"]].values
    clf.fit(data)
    all["where"]=pd.Series(clf.labels_)

    #经理平均每天工作多少小时
    all["all_hours"]=all["time"]*24+all["created_hour"]
    #这个manager总共在多少点上发了信息，这个经理勤奋吗
    add = pd.DataFrame(all.groupby(["manager_id"]).all_hours.nunique()).reset_index()
    add.columns = ["manager_id", "manager_hours"]
    all = all.merge(add, on=["manager_id"], how="left")
    all["manager_hours_rt"]=all["manager_hours"]/all["manager_active"]

    #为了顺序占位
    all["manager_price_mean"]=0

    #经理总共发出多少price的listing
    add = pd.DataFrame(all.groupby(["manager_id"]).price.sum()).reset_index()
    add.columns = ["manager_id", "manager_price_sum"]
    all = all.merge(add, on=["manager_id"], how="left")

    #经理总共发出多少个卧室的listing
    add = pd.DataFrame(all.groupby(["manager_id"]).bedrooms.sum()).reset_index()
    add.columns = ["manager_id", "manager_bedrooms_sum"]
    all = all.merge(add, on=["manager_id"], how="left")

    #经理总共挣了多少钱
    add = pd.DataFrame(all.groupby(["manager_id"]).building_dif.sum()).reset_index()
    add.columns = ["manager_id", "earn_all"]
    all = all.merge(add, on=["manager_id"], how="left")

    #经理平均每个bedroom挣多少钱
    all["manager_price_mean"]=all["manager_price_sum"]/all["manager_bedrooms_sum"]

    #经理平均每天挣多少钱
    all["earn_everyday"]=all["earn_all"]/all["manager_active"]

    #是在多大的交易量下挣的这些钱(投资回报比)
    all["earn_all_rt"]=all["earn_all"]/all["manager_price_sum"]

    #平均每天的发布额
    all["manager_price_"] = all["manager_price_sum"] / all["manager_active"]

    #经纬度附近的房子中价格比这个低的个数（需要优化）
    #暂时调用结果测试
    neak=pd.read_csv("timeout.csv")
    aaaa=neak[["jwd_type_low_than_num","jwd_type_all","jwd_type_rt","listing_id"]]
    all=all.merge(aaaa,on="listing_id",how="left")
    #all["jwd_type_low_than_num"]=map(lambda lo,la,ba,be,p:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)&(all.price<=p)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"],all["price"])
    #all["jwd_type_all"]=map(lambda lo,la,ba,be:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"])
    #all["jwd_type_rt"]=all["jwd_type_low_than_num"]/all["jwd_type_all"]

    #用低于该价格多少来表示经理开价程度
    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_type_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay_jwd_type_rt"]
    all = all.merge(add, on=["manager_id"], how="left")

    #这个放在这只是为了复现顺序。。。
    #开启五个聚类的特征构造
    where_mean={}
    where_list=list(all["where"].value_counts().index)
    for w in where_list:
        where_mean[w]=all[all["where"]==w].price.mean()
    print where_mean
    all["where_mean"]=map(lambda x:where_mean[x],all["where"])
    all["where_mean_rt"]=all["price"]/all["where_mean"]

    #经理的活动范围距离市区多远
    add = pd.DataFrame(all.groupby(["manager_id"]).distance.mean()).reset_index()
    add.columns = ["manager_id", "manager_distance"]
    all = all.merge(add, on=["manager_id"], how="left")

    #经理发帖时间集中在哪里
    add = pd.DataFrame(all.groupby(["manager_id"]).created_hour.var()).reset_index()
    add.columns = ["manager_id", "manager_post_hour_var"]
    all = all.merge(add, on=["manager_id"], how="left")

    #经理发帖时间稳定性
    add = pd.DataFrame(all.groupby(["manager_id"]).created_hour.mean()).reset_index()
    add.columns = ["manager_id", "manager_post_hour_mean"]
    all = all.merge(add, on=["manager_id"], how="left")

    #放在这里完全是为了顺序。。。
    #添加四个均价和距离的关系，添加了5的平滑系数
    all["manager_price_distance_rt"]=all["manager_price_mean"]/(all["manager_distance"]+5)
    all["fangxing_mean_distance_rt"]=all["fangxing_mean"]/(all["distance"]+5)
    all["building_mean_distance_rt"]=all["building_mean"]/(all["distance"]+5)
    all["price_mean_jwd_distance_rt"]=all["price_mean_jwd"]/(all["distance"]+5)

    all["man_bui_id"]=map(lambda x,y:str(x)+str(y),all["manager_id"],all["building_id"])
    all["price_bath_bed"] = all["price"]/(all["bathrooms"]/2.0 + all["bedrooms"]+1)  #重构，覆盖hello里的结果

    #将特征分配到每个manager，取前几个最大次数的
    #"""
    manager_list = list(all["manager_id"].value_counts().index)
    manager_feature={}
    for man in manager_list:
        content = []
        for i in all[all.manager_id==man]['features']:
            content.extend(i.lower().replace("[","").replace("]","").replace("-","").replace("/","").replace(" ","").split(","))
        abc = pd.Series(content).value_counts()
        new=list(abc.index)[:20]
        try:
            feature=",".join(new)
        except:
            feature=""
        manager_feature[man]=feature+","
    all["manager_features"]=map(lambda x:manager_feature[x],all["manager_id"])

    # 平均每天发多少
    all["post_day"] = all["manager_count"] / all["manager_active"]

    #给feature打分
    all["features"]=all["features"].apply(lambda x:x.lower().replace("[","").replace("]","").replace("-","").replace("/","").replace(" ",""))

    content = []
    for i in all[(all.interest_level=="high")|(all.interest_level=="medium")]["features"]:
        if i != "":
            content.extend(i.split(","))
    good = pd.Series(content).value_counts().to_frame(name="num_good")
    content = []
    for i in all[(all.interest_level=="low")]["features"]:
        if i != "":
            content.extend(i.split(","))
    bad = pd.Series(content).value_counts().to_frame(name="num_bad")
    tongji=good.merge(bad, left_index=True, right_index=True,how="outer").fillna(0)#iloc[0:200]
    abc=tongji["num_good"]/(tongji["num_bad"]+1)
    def score(x):
        score=0
        for i in x.split(","):
            try:
                score+=abc[i]
            except:
                pass
        return score
    all["manager_feature_score"]=map(lambda x:score(x),all["manager_features"])

    #manager和price_bath_bed的关系
    add = pd.DataFrame(all.groupby(["manager_id"]).price_bath_bed.mean()).reset_index()
    add.columns = ["manager_id", "manager_price_bath_bed_mean"]
    all = all.merge(add, on=["manager_id"], how="left")

    #manager和房子中building_id为0的关系
    manager_building_zero_count={}
    for man in manager_list:
        manager_building_zero_count[man]=all[(all.manager_id==man)&(all.building_id.astype("str")=="0")].shape[0]
    all["manager_building_zero_count"]=map(lambda x:manager_building_zero_count[x],all["manager_id"])
    all["manager_building_zero_count_rt"]=all["manager_building_zero_count"]/all["manager_count"]

    #manager的经纬度中位数
    add = pd.DataFrame(all.groupby(["manager_id"]).longitude.median()).reset_index()
    add.columns = ["manager_id", "manager_longitude_median"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).latitude.median()).reset_index()
    add.columns = ["manager_id", "manager_latitude_median"]
    all = all.merge(add, on=["manager_id"], how="left")

    #除价格其他都一样的个数
    all["same"]=map(lambda a,b,c,d,e:str(a)+str(b)+str(c)+str(d)+str(e),all["manager_id"],all["bedrooms"],all["bathrooms"],all["building_id"],all["features"])
    same_count = all["same"].value_counts()
    all["same_count"] = map(lambda x: same_count[x], all["same"])

    #每个经理每个building的房子数
    man_bui_id_count = all["man_bui_id"].value_counts()
    all["man_bui_id_count"] = map(lambda x: man_bui_id_count[x], all["man_bui_id"])
    #每个经理拥有该building房子数的比例
    all["man_bui_id_count_rt"] = all["man_bui_id_count"]/all["building_count"]

    #房间面积
    all["acreage"]=1+all["bedrooms"] + all["bathrooms"]/2.0

    #街区距离
    all["jq_distance"] = map(lambda x, y: (abs(x - b)+abs(y-a))*111, all["longitude"],all["latitude"])

    #该区域内的listing数
    add = pd.DataFrame(all.groupby(["jwd_class"]).listing_id.count()).reset_index()
    add.columns = ["jwd_class", "listing_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    #平均每个房子租多少间出去
    all["building_listing_num_jwd_rt"]=all["building_num_jwd"]/all["listing_num_jwd"]

    #添加地点的斜率,和前面的distance一起唯一确定该点和选取的点之间的相对位置
    all["lo_la"] = (all["longitude"]-b) / (all["latitude"]-a)

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
    aaaa=neak[["listing_id","building_zero_num"]]
    all=all.merge(aaaa,on="listing_id",how="left")
    #all["building_zero_num"] = map(lambda la, lo: building_zero_num(la, lo,1), all["latitude"], all["longitude"])

    #添加leak
    print "添加图片leak"
    time_stamp=pd.read_csv("listing_image_time.csv")
    all=all.merge(time_stamp,on="listing_id")

    #随机选取有一定间隔的六个点，类似于指定聚类中心
    la1, lo1 =40.778772,-73.96684
    la2, lo2=40.849209,-73.888508
    la3, lo3 =40.747844,-73.901731
    la4, lo4 =40.678722,-73.951174
    la5, lo5 =40.688788,-73.870111
    la6, lo6 =40.624861,-73.967846
    all["dis_1"]=map(lambda la,lo:abs(la-la1)+abs(lo-lo1),all["latitude"],all["longitude"])
    all["dis_2"]=map(lambda la,lo:abs(la-la2)+abs(lo-lo2),all["latitude"],all["longitude"])
    all["dis_3"]=map(lambda la,lo:abs(la-la3)+abs(lo-lo3),all["latitude"],all["longitude"])
    all["dis_4"]=map(lambda la,lo:abs(la-la4)+abs(lo-lo4),all["latitude"],all["longitude"])
    all["dis_5"]=map(lambda la,lo:abs(la-la5)+abs(lo-lo5),all["latitude"],all["longitude"])
    all["dis_6"]=map(lambda la,lo:abs(la-la6)+abs(lo-lo6),all["latitude"],all["longitude"])

    all["class_lo_la"]=np.argmin(all[["dis_1","dis_2","dis_3","dis_4","dis_5","dis_6"]].values,axis=1)
    all["class_lo_la_dis"]=np.min(all[["dis_1","dis_2","dis_3","dis_4","dis_5","dis_6"]].values,axis=1)

    #添加每个listing的图片的平均大小
    import json
    with open("jpgs.json", "r") as f:
        data = f.read()

    data = json.loads(data)
    img_dic = {}
    for i in data.keys():
        img_list = data[i]
        shape_list = []
        for img in img_list:
            shape = img[0] * img[1]
            shape_list.append(shape)
        leng = len(img_list)
        try:
            img_dic[int(i)] = sum(shape_list) / leng
        except:
            img_dic[int(i)] = 0

    all["pic_mean"]=map(lambda x:img_dic.get(x,0),all["listing_id"])

    #用gdy的manager表征
    train_add=pd.read_csv("train_gdy.csv")
    test_add=pd.read_csv("test_gdy.csv")
    add=train_add.append(test_add)
    all=all.merge(add,on="listing_id",how="left")

    ######################
    #继续添加特征：
    all["feature_price_rt"]=all["price"]/all["feature_num"]
    all["photo_price_rt"]=all["price"]/all["photo_num"]

    price_today=pd.DataFrame(all.groupby(["time"]).price.median()).reset_index()
    price_today.columns=["time","price_today"]
    all=all.merge(price_today,on="time",how="left")

    price_created_month=pd.DataFrame(all.groupby(["created_month"]).price.median()).reset_index()
    price_created_month.columns=["created_month","price_today"]
    all=all.merge(price_created_month,on="created_month",how="left")

    all["price_rt_jwd"] = all["price"] / all["type_jwd_price_mean"]

    all.to_csv("all20.csv",index=None)
    #处理离散
    addclass=["man_bui_id",]
    categorical = ["display_address", "manager_id", "building_id", "street_address"]+addclass
    # categorical = ["display_address","manager_id", "building_id"]
    for f in categorical:
        if all[f].dtype == 'object':
            # print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(all[f].values))
            all[f] = lbl.transform(list(all[f].values))

    all=all.replace({"interest_level":{"high":0,"medium":1,"low":2,"nnnn":3},
                     "description":{0:"o"}
                     })
    train = all[all.interest_level != 3].copy()
    valid = all[all.interest_level == 3].copy()
    y_train=train["interest_level"]

    train_num=train.shape[0]

    tfidf = CountVectorizer(stop_words='english', max_features=100)
    all_sparse=tfidf.fit_transform(all["features"].values.astype('U'))
    tr_sparse = all_sparse[:train_num]
    te_sparse = all_sparse[train_num:]
    #print tfidf.get_feature_names()

    x_train = train.drop(["interest_level","features","description","manager_features","same"],axis=1)
    x_valid = valid.drop(["interest_level","features","description","manager_features","same"],axis=1)

    x_train = sparse.hstack([x_train.astype(float),tr_sparse.astype(float)]).tocsr()
    x_valid = sparse.hstack([x_valid.astype(float),te_sparse.astype(float)]).tocsr()

    return x_train,y_train,x_valid,valid

def run(train_matrix,test_matrix):
    params = {'booster': 'gbtree',
              #'objective': 'multi:softmax',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'gamma': 1,
              'min_child_weight': 1.5,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.03,
              'tree_method': 'exact',
              'seed': 2017,
              'nthread': 12,
              "num_class":3
              }
    num_round = 10000
    early_stopping_rounds = 50
    watchlist = [(train_matrix, 'train'),
                 (test_matrix, 'eval')
                 ]
    if test_matrix:
        model = xgb.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                      early_stopping_rounds=early_stopping_rounds
                      )
        pred_test_y = model.predict(test_matrix,ntree_limit=model.best_iteration)
        return pred_test_y, model
    else:
        model = xgb.train(params, train_matrix, num_boost_round=num_round
                      )
        return model


def XGB():


    #X, y = get_data()
    """
    train_x=X[:10000,:]
    test_x=X[10000:,:]
    train_y=y[:10000]
    test_y=y[10000:]
    """
    X,y,z,v = get_data()
    print X.shape

    """
    X=all_X[:30000,:]
    v_X=all_X[30000:,:]
    y=all_y[:30000]
    v_y=all_y[30000:]
    """


    #V=xgb.DMatrix(v_X,label=v_y)
    z = xgb.DMatrix(z)

    #print X.shape
    #print z.shape
    #train_x=X[:40000]
    #test_x=X[40000:]

    #train_y=y[:40000]
    #test_y=y[40000:]

    #train_matrix = xgb.DMatrix(X, label=y)
    cv_scores = []
    model_list=[]
    preds_list=[]
    kf = cross_validation.KFold(X.shape[0],n_folds=5,shuffle=True,random_state=1)
    for dev_index, val_index in kf:
        train_x, test_x = X[dev_index, :], X[val_index, :]
        train_y, test_y = y[dev_index], y[val_index]
        train_matrix = xgb.DMatrix(train_x, label=train_y,missing=-1)
        test_matrix = xgb.DMatrix(test_x, label=test_y,missing=-1)
        preds, model = run(train_matrix, test_matrix)
        cv_scores.append(log_loss(test_y, preds))
        model_list.append(model)
        preds_list.append(preds)
        print cv_scores
        with open("result.txt","a") as f:
            f.write(str(cv_scores)+"\n")
        #break
    #组装preds
    for i in range(len(preds_list)):
        if i==0:
            pre=preds_list[i]
            pre_v=model_list[i].predict(z,ntree_limit=model.best_iteration)
        else:
            pre=np.concatenate((pre,preds_list[i]),axis=0)
            pre_v=(pre_v+model_list[i].predict(z,ntree_limit=model.best_iteration))

    pre_v=pre_v/len(preds_list)

    loss_mean=np.mean(cv_scores)
    print loss_mean
    with open("result.txt", "a") as f:
        f.write(str(loss_mean) + "\n")

    result=pre_v
    out_df = pd.DataFrame(result)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = v.listing_id.values
    out_df.to_csv("xgb_cv10_%s.csv" % str(loss_mean), index=False)

    """
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    print importance
    """
    for model in model_list:
        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        print importance


XGB()
"""
X, y, z, v = get_data()
print X.shape
z = xgb.DMatrix(z)
train_matrix = xgb.DMatrix(X, label=y,missing=-1)
model=run(train_matrix,"")
result = model.predict(z,ntree_limit=model.best_iteration)
out_df = pd.DataFrame(result)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = v.listing_id.values
out_df.to_csv("xgb_single_eta0.01.csv", index=False)
"""