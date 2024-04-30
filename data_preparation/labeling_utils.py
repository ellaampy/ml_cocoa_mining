import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import rasterio as rio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
import joblib
import os
from tqdm import tqdm


def zonal(shp_path, raster_path, stats='mean', col_prefix='sentinel'):
    """
    args
    ----
        shp_path: path to shapefile or shp instance
        raster_path: path to .tif file
        stats: statistic operation to compute 
               e.g mean, median
        col_prefix: column prefix for raster band 
               e.g. sentinel_2, dem_1

    returns
    -------
        shapefile with zonal statistics
    
    """
    if isinstance(shp_path, str):
        shp = gpd.read_file(shp_path)
    else:
        shp = shp_path

    # iterate raster and band
    raster = rio.open(raster_path)

    # get statistic per segment
    band_stats = {}
    for i in range(1, raster.count + 1):

        band = raster.read(i)

        # Calculate zonal statistics
        result = zonal_stats(shp, band, stats=stats, nodata=-9999, \
                             all_touched=True, affine=raster.transform)

        # normalize (divide by 10000 for sentinel and 500 for dem)
        # dem max value for area of interest is 500

        if col_prefix == 'sentinel':
            result = [val[stats]/10000 for val in result]

        elif col_prefix == 'dem':
            result = [val[stats]/500 for val in result]
        
        band_stats[col_prefix+str(i)] = result

    # join results to shp
    shp = shp.assign(**band_stats)

    return shp



def model_train(table, splits, num_trial, n_jobs, model_path):

    """
    args
        table : pandas df object
        splits : number of k-fold
        num_trial: number of optuna trials. if 0, default classifier used
        n_jobs: number of jobs
        model_path: directory+path.joblib to save model
    """

    # remove geometry and FID
    cols = [x for x in table.columns if x not in ['FID', 'geometry']]
    table = table[cols]

    y = table['Target'].values
    X = table.drop(columns=['Target']).values

    # Setting up the stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    # List to store metrics of each fold
    f_scores = []
    accuracies = []  
    fold_num = 0

    # 5-fold Stratified Cross Validation loop
    for train_index, test_index in skf.split(X, y):
        fold_num += 1

        # Splitting the dataset for this fold
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train_labels, y_test_labels = [y[i] for i in train_index], [y[i] for i in test_index]
        
        # activate tuning for single fold
        if num_trial > 0 and fold_num == 1:  

            def objective(trial):
                params = {
                'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
                'max_depth' : trial.suggest_int('max_depth', 3, 15),
                'min_samples_split' : trial.suggest_int('min_samples_split', 5, 32),
                'bootstrap' : trial.suggest_categorical('bootstrap', [True, False]),
                'n_jobs' : trial.suggest_categorical('n_jobs', [n_jobs]),
                'random_state' : trial.suggest_categorical('random_state', [42])
                }

                clf = RandomForestClassifier(**params)
                clf.fit(X_train, y_train_labels)

                # Making predictions on the test set
                y_pred = clf.predict(X_test)

                # Calculating and reporting the accuracy
                accuracy = f1_score(y_test_labels, y_pred)
                return accuracy


            # optimize study
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=num_trial) 
            print('best study parameters', study.best_params)
        

        # Model training
        if num_trial > 0:
            clf = RandomForestClassifier(**study.best_params)
        else:
            clf = RandomForestClassifier(n_estimators=1000, n_jobs=n_jobs)

        clf.fit(X_train, y_train_labels)
        
        # Making predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculating and reporting the accuracy
        accuracy = accuracy_score(y_test_labels, y_pred)
        accuracies.append(accuracy)  # Storing the accuracy
        
        # Calculating and reporting the fscore
        f_score = f1_score(y_test_labels, y_pred, average='weighted')
        f_scores.append(f_score)  # Storing the accuracy

    # save model
    joblib.dump(clf, model_path)

    # Reporting the final results
    avg_accuracy = np.mean(accuracies)
    avg_fscore = np.mean(f_scores)
    print(f"Average Accuracy across all folds: {avg_accuracy:.4f}")
    print(f"Average Fscore across all folds: {avg_fscore:.4f}")

    return clf, accuracies, f_scores


def model_pred(shp_path, sentinel_path, dem_path, model_path, output_shp, return_prob=True):

    shp = zonal(shp_path, sentinel_path, stats='mean', col_prefix='sentinel')
    shp = zonal(shp, dem_path, stats='mean', col_prefix='dem')

    # get existing attributes
    FID = shp['FID'].tolist()
    geom = shp['geometry'].tolist()
    probs = []

    # remove geometry, FID, target columns
    cols = [x for x in shp.columns if x not in ['FID', 'geometry', 'Target']]
    shp = shp[cols]

    
    # prediction block
    X = shp.values
    clf = joblib.load(model_path)

    for i in tqdm(range(X.shape[0])):
        X_subset = X[i].reshape(1, -1)

        if return_prob:
            predictions = clf.predict_proba(X_subset)
        else:
            predictions = clf.predict(X_subset)
        
        # get prob of mining class (label 1)
        probs.append(predictions[:,1][0])

    
    # save predictions to model path
    predictions_gdf = gpd.GeoDataFrame(geometry=geom, crs='epsg:32630')
    predictions_gdf['FID'] = FID
    predictions_gdf['mine_prob'] = probs

    # Extract directory and file name
    output_dir, output_file = os.path.split(output_shp)
    
    # Ensure the directory exists before saving the file
    if output_dir and not os.path.exists(output_dir):
        #otherwise create the dir
        os.makedirs(output_dir)

    # Save GeoDataFrame to file
    predictions_gdf.to_file(output_shp)

    print('prediction complete')
    return predictions_gdf




# ## how to use

# # load labeled shp (e.g the labeled portion of cluster 0)
# labeled_shp_path = '/app/dev/FM4EO/testing/samples_2000.shp'
labeled_shp_path = '/localhome/zapp_an/Desktop/fasteo/data_2016/cluster_3_labels/clustered_polygons_3_labels.shp'

# # load unlabeled shp eg. cluster 0 shp
# unlabeled_shp_path = '/app/dev/FM4EO/testing/clustered_polygons_0.shp'
unlabeled_shp_path = '/localhome/zapp_an/Desktop/fasteo/data_2016/cluster_3/clustered_polygons_3.shp'

# # path to save predicted labels of unlabeled shp
# predicted_output= '/app/dev/FM4EO/testing/clustered_polygons_0_labeled.shp'
predicted_output= '/localhome/zapp_an/Desktop/fasteo/data_2016/cluster_2_predictions/predictions_cluster_2_2016.shp'

# # path to sentinel2 image and dem
# raster_path = '/app/dev/FM4EO/data/mosaic/mosaic_2016_final.tif'
# dem_path = '/app/dev/FM4EO/data/cop_dem/elevation.tif'
raster_path = '/localhome/zapp_an/Desktop/fasteo/data_2016/mosaic_2016_final.tif'
dem_path = '/localhome/zapp_an/Desktop/fasteo/ml_cocoa_mining/data_preparation/elevation.tif'

# # path to save rf model
# model_path = '/app/dev/FM4EO/testing/rf_2016_cluster0.joblib'
model_path = '/localhome/zapp_an/Desktop/fasteo/ml_cocoa_mining/data_preparation/rf_2016_cluster2.joblib'


# # get zonal statistics of sentinel bands and dem for labeled shp
shp_original = zonal(labeled_shp_path, raster_path, stats='mean', col_prefix='sentinel')
shp_original = zonal(shp_original, dem_path, stats='mean', col_prefix='dem')

# # train random forest on labeled. returns model accuracy, fscore
model, acc, fs = model_train(shp_original, splits=3, num_trial= 50, n_jobs=30, model_path=model_path)

# # predict on unlabeled
model_pred(unlabeled_shp_path, raster_path, dem_path, model_path, predicted_output, return_prob=True)