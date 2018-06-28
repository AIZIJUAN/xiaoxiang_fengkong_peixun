ROOT_DIR = 'D:/PythonProject/A_Scorecard/'

NUM_BINS = 5

num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc','limit_income','earliest_cr_to_app']
#类别型变量
cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']


xgb_params = {
        'min_child_wight' : 100,
        'eta' : 0.02,
        'colsample_bytree' : 0.8,
        'max_depth' : 6,
        'subsample' : 0.8,
        'alpha' : 1,
        'gamma' : 1,
        'slient' : 1,
        'verbose_eval' : True,
        'seed' : 1024
    }
