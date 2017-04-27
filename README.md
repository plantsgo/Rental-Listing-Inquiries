# Rental-Listing-Inquiries
kaggle Two Sigma Connect: Rental Listing Inquiries

After learned solutions by others,especially Little Boat,simple but effective!!amazing!
I spend a lot of time FE,and submit more times then Faron,Little Boat et.
I'll thanks KazAnova who public the leak,otherwise I can't do this better.You are my idol.
My English is poor,if I write confused,please tell me,many thanks.

FE:
I split the base features two classes:
1:manager:created,description,price,et
2:building:bathrooms,bedrooms,latitude,longitude,display_address,featuers,photos,et
then I link and compare them.

My best single model at: https://github.com/plantsgo/Rental-Listing-Inquiries

score:Public:0.50379  Private:0.50500

you should add train.json and test.json in the folder.
the jpgs.json is the shape of each photos.
the listing_image_time.csv is the leak @KazAnova said.

1.run sigma.py to create the csv file.
2.run script.py to create the features what @gdy5 show.
3.run feature_tt.py to create the base features.
4.run feature_tt_long.py to create four features which spend long time,about four hours...but I have give it which named timeout.csv,so you can skip it.
5.run xgb.py and will create the last result.

Transform：
1.X
2.log10（X+1)
My best nn model is log10(X+1) score LB:0.535 before add magic feature.

Ensemble:
Level 2:
I have 4 datasets.
1.My best single model.
2.some features which not improve at my best model ,but can improve at model with base features.
3.@gdy5 's kernel with some of my features.
4.@Branden Murrayit 's kernel add some of my features.

①:each dateset I used [xgb,nn,gb,rf,et,lr,xgb_reg,lgb_reg,nn_reg] cv flod=5
the reg model have a good importance in my model.

②:and I merge high and medium level ,then userd[lgb,nn,lgb_reg,nn_reg,et,rf] in my best dataset. cv flod=5

③:[nn,nn_reg,xgb,gb,rf,et,lr,xgb_reg]@last three datasets   cv flod=5

④:[nn,nn_reg,xgb,gb,rf,et,lr,xgb_reg]add magic feature @last three datasets   cv flod=5

⑤:[nn,nn_reg,xgb,knn,gb,rf,et,lr,ada_reg,rf_reg,gb_reg,et_reg,xgb_reg]@last three datasets   cv flod=10


Level 3:
1.user ①,②,③,④ as metefeatures with xgb,nn,et.
with a feature from description.
classify the source by description:
        
CooperCooper.com
<p><a  website_redacted
</li></ul></p>

pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15
then userd @weiwei 's Prior correction. but only improved 0.00001-0.00002

2.user ①,②,⑤ as metefeatures with xgb,nn,et.
pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15

Level 4:
50/50 average level 3


Last,Thanks all shares,I learned many from the kernels and discussions.









