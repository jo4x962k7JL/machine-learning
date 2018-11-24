
### Install

This project requires Python and the following Python libraries installed:
[NumPy](http://www.numpy.org/)、[Pandas](http://pandas.pydata.org/)、[matplotlib](https://matplotlib.org/)、[scikit-learn](http://scikit-learn.org/stable/)、[LightGBM](https://github.com/Microsoft/LightGBM)、[CatBoost](https://tech.yandex.com/catboost/)、[XGBoost](https://xgboost.readthedocs.io/en/latest/)、[Kaggle API](https://github.com/Kaggle/kaggle-api)

### Run
1. **Download dataset** from Kaggle and save in dataset/

```bash
kaggle competitions download -c home-credit-default-risk
```

2. **One-line command** in CLI

```bash
bash ./run.sh
```

3. **Submit to Kaggle** and get AUC=0.79521 on private LB
(Ranking 535th out of 7,198 teams, **TOP 8%**)

```bash
kaggle competitions submit home-credit-default-risk -f report.csv -m 'I love you, jo4x962k7JL'
kaggle competitions submissions -c home-credit-default-risk
```

