
# Kaggle Rethop Solution Ranking 1st

## score: *Public*:0.50379    *Private*:0.50500

## Features Enginnering

I split the base features two classes:

1. manager:created,description,price,et

2. building:bathrooms,bedrooms,latitude,longitude,display_address,featuers,photos,et

## Files 

* *the jpgs.json* is the shape of each photos.

* *listing_image_time.csv* is the leak @KazAnova said.

## How to Use

1. *sigma.py* to create the csv file.

2. *script.py* to create the features what @gdy5 show.

3. *feature_tt.py* to create the base features.

4. *feature_tt_long.py* to create four features which spend about four hours...but I have generate already which named timeout.csv,so you can skip it....

5. *xgb.py* and will create the last result.

## Transform

  1. X    

  2. log10（X+1)    

## Ensemble

  My best nn model is log10(X+1) score *LB:0.535* before add magic feature.

### Level 2:

I have 4 datasets:

  1. My best single model.

  2. some features which not improve at my best model ,but can improve at model with base features.

  3. @gdy5 's kernel with some of my features.

  4. @Branden Murrayit 's kernel add some of my features.


Each dateset I used [xgb,nn,gb,rf,et,lr,xgb_reg,lgb_reg,nn_reg] cv fold=5

the reg model have a good importance in my model.


### Level 3:

  1. 1,2,3 metefeatures with xgb,nn,et.

  *pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15*
  then userd @weiwei 's Prior correction. but only improved 0.00001-0.00002

  2. 1,2,3,4 metefeatures with xgb,nn,et.
  *pre=((xgb^0.65)*(nn^0.35))*0.85+et*0.15*

### Level 4:
  50/50 average level 3


##*Last,Thanks all shares,I learned many from the kernels and discussions.*










