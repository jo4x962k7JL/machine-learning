echo '=============== Start preprocess POS_CASH_balance ==============='
python3 preprocessing_POS_CASH_final.py
echo '=============== Start preprocess application_train and application_test ==============='
python3 preprocessing_application_final.py
echo '=============== Start preprocess bureau and bureau_balance ==============='
python3 preprocessing_bureau_final.py
echo '=============== Start preprocess credit card balance ==============='
python3 preprocessing_credit_final.py
echo '=============== Start preprocess installments_payments ==============='
python3 preprocessing_install_final.py
echo '=============== Start preprocess previous_applications ==============='
python3 preprocessing_previous_final.py
echo '=============== Start generate a whole new file for training and predicting ==============='
python3 datagen.py
echo '=============== Start Training 1st-level(3 models) and 2nd-level model ==============='
python3 report.py
echo '==================================================='
echo ' END '
echo '==================================================='
