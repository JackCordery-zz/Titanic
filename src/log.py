import pandas as pd

def get_n_features(name):
    return name.split("-")[1]

def get_model_name(name):
    return name.split("-")[0]

def log_hyper_tuning(stats):
    best_parameters = stats["best_parameters"]
    scores = stats["scores"]
    full_results = stats["full_results"]

    df_parameters = pd.DataFrame.from_dict(best_parameters, orient='index')
    df_scores = pd.DataFrame.from_dict(scores, orient='index')
    df_scores.columns = ["score"]
    df = pd.concat([df_parameters, df_scores], axis=1)

    full_results = {name: df_temp.sort_values(["mean_test_accuracy_score",
                                               "mean_test_auc_score"],
                                               ascending=False)
                    for name, df_temp in full_results.items()}

    return df, full_results

def log_feature_selection(stats):
    all_features = stats["all_features"]
    support = stats["support"]
    score_mean = stats["score_mean"]
    score_std = stats["score_std"]

    df_support = pd.DataFrame.from_dict(support, orient='index')
    df_support.columns = all_features
    
    df_mean = pd.DataFrame.from_dict(score_mean, orient='index')
    df_mean.columns = ["mean"]

    df_std = pd.DataFrame.from_dict(score_std, orient='index')
    df_std.columns = ["std"]

    df = pd.concat([df_support,df_mean,df_std], axis=1)
    df.reset_index(inplace=True)
    df["n_features"] = df["index"].apply(get_n_features)
    df["index"] = df["index"].apply(get_model_name)
    df.sort_values(["mean", "n_features"], ascending=[False, True], inplace=True)
    
    return df

def log_results(feature_stats, hyper_stats):
    f = log_feature_selection(feature_stats)
    h, r = log_hyper_tuning(hyper_stats)
    return f, h, r

def main():
    print("Nothing to see here: Log")
    return

if __name__ == '__main__':
    main()