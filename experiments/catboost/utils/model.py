import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from catboost.utils import get_gpu_device_count
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score


###   catboost   ###
def model_catboost(X_train, y_train, X_test, y_test, mode, group_size):
    def model_binary(X_train, y_train, X_test, y_test):
        gpu_count = get_gpu_device_count()
        clf = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',   # 標準的 ROC AUC
            iterations=10000,
            # learning_rate=0.001,
            random_seed=42,
            verbose=100,
            l2_leaf_reg=7,
            task_type='GPU' if gpu_count > 0 else 'CPU',
        )

        clf.fit(X_train, y_train,
                use_best_model=True,
                eval_set=(X_test, y_test),
                early_stopping_rounds=300,
            )
        return clf

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_multiary(X_train, y_train, X_test, y_test):
        gpu_count = get_gpu_device_count()
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='AUC',  # Micro-averaged One-vs-Rest ROC AUC
            iterations=10000,
            learning_rate=0.007,
            random_seed=42,
            verbose=100,
            boosting_type='Ordered',
            l2_leaf_reg=7,
            task_type='GPU' if gpu_count > 0 else 'CPU',
        )
        clf.fit(X_train, y_train,
                use_best_model=True,
                eval_set=(X_test, y_test),
                early_stopping_rounds=300,
            )
        return clf
    if mode == "binary":
        return model_binary(X_train, y_train, X_test, y_test)
    elif mode == "multiary":
        return model_multiary(X_train, y_train, X_test, y_test)
    

###   random_forest   ###
def model_random_forest(X_train, y_train, X_test, y_test, mode, group_size):
    def model_binary(X_train, y_train, X_test, y_test):
        clf = RandomForestClassifier(
            n_estimators=1500,          # 樹的數量
            max_depth=10,               # 樹的深度
            min_samples_split=4,        # 最小分裂樣本數
            min_samples_leaf=2,         # 最小葉節點樣本數
            class_weight='balanced',    # 類別權重
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # param_dist = {
        #     'n_estimators': randint(500, 2000),
        #     'max_depth': randint(5, 20),
        #     'min_samples_split': randint(2, 10),
        #     'min_samples_leaf': randint(1, 5),
        #     'class_weight': ['balanced', None]
        # }
        # base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        # search = RandomizedSearchCV(
        #     estimator=base_model,
        #     param_distributions=param_dist,
        #     n_iter=20   ,
        #     scoring='roc_auc',
        #     cv=3,
        #     random_state=42,
        #     verbose=2,
        #     n_jobs=-1
        # )

        # search.fit(X_train, y_train)
        # print("[Best Parameters Found for Binary Classification]:")
        # print(search.best_params_)
        # best_rf = search.best_estimator_
        # best_rf.fit(X_train, y_train)

        return clf

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_multiary(X_train, y_train, X_test, y_test):
        clf = RandomForestClassifier(
            n_estimators=2500,        # 很多樹，保證穩定（適合多類別）
            max_depth=22,             # 深一點，因為多類別邊界複雜
            min_samples_split=5,      # 每個節點至少 5 筆資料才能分裂
            min_samples_leaf=2,       # 葉子節點至少留 2 筆資料
            max_features='sqrt',      # 每次分裂考慮 log2(特徵數)個特徵，防止過擬合
            random_state=42,
            n_jobs=-1
            )


        clf.fit(X_train, y_train)
        return clf
    
    if mode == "binary":
        return model_binary(X_train, y_train, X_test, y_test)
    elif mode == "multiary":
        return model_multiary(X_train, y_train, X_test, y_test)
    

def model(X_train, y_train, X_test, y_test, mode="binary", model_str="random_forest", group_size=27):
    if model_str == "catboost":
        model_train = model_catboost(X_train, y_train, X_test, y_test, mode, group_size)
    elif model_str == "random_forest":
        model_train = model_random_forest(X_train, y_train, X_test, y_test, mode, group_size)
    # elif model_str == "xgboost":
    #     model_train = model_xgboost(X_train, y_train, X_test, y_test, mode, group_size)
    else:
        raise ValueError("Invalid model type. Choose 'catboost', 'random_forest', or 'xgboost'.")
    
    if mode == "binary":
        predicted = model_train.predict_proba(X_test)
        # 取出正類（index 0）的概率
        predicted = [predicted[i][0] for i in range(len(predicted))]
        
        num_groups = len(predicted) // group_size 
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        y_pred  = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f'Binary AUC of {model_str.capitalize()}:',auc_score, "\n")
    elif mode == "multiary":
        predicted = model_train.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            # 對每個類別計算該組內的總機率
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print(f'Multiary AUC of {model_str.capitalize()}:', auc_score, "\n")

    else:
        raise ValueError("Invalid mode. Choose 'binary' or 'multiary'.")
    return model_train