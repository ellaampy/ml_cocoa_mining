import os
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
import re
import pandas as pd
from joblib import dump, load
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from imblearn.under_sampling import RandomUnderSampler


def merge_csv(base_dir):
    """Load and merge csv containing train/test split"""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                data = pd.read_csv(os.path.join(root, file))
                csv_files.append(data)

    df = pd.concat(csv_files)

    # seperate into train and test split
    df_train = df[df['split']=='train']['patch_name'].to_list()
    df_test = df[df['split']=='test']['patch_name'].to_list()

    return df_train, df_test


def load_data_by_splits(base_dir, splits, bands):
    features = []
    labels = []
    
    for file in splits:
        match = re.search(r"_(\d{4})\.tif$", file)
        year = match.group(1)
        img_path = os.path.join(base_dir, str(year), 'IMAGE', file.replace('MASK', 'IMG'))
        mask_path = os.path.join(base_dir, str(year), 'MASK', file)

        img_data = load_tiff_as_array(img_path)
        mask_data = load_tiff_as_array(mask_path).squeeze()  # Remove singleton band dim
        img_flat = img_data.reshape(-1, img_data.shape[-1])  # Flatten HxWxB to NxB
        mask_flat = mask_data.flatten()  # Flatten HxW to N
        features.append(img_flat[:, bands])
        labels.append(mask_flat)

    return np.vstack(features), np.hstack(labels)
    


def predict_patches(base_dir, splits, rf_model, bands, results_dir):
    ids = []
    predictions = []
    ground_truth = []
    # miou = []
    # acc = []

    for file in tqdm(splits):
        match = re.search(r"_(\d{4})\.tif$", file)
        year = match.group(1)
        img_path = os.path.join(base_dir, str(year), 'IMAGE', file.replace('MASK', 'IMG'))
        mask_path = os.path.join(base_dir, str(year), 'MASK', file)

        img_data = load_tiff_as_array(img_path)
        mask_data = load_tiff_as_array(mask_path).squeeze()  # Remove singleton band dim

        # slice bands
        img_data = img_data[:, :, bands]
        pred = rf_model.predict(img_data.reshape(-1, img_data.shape[-1]))
        predictions.append(pred.reshape(128,128))
        ground_truth.append(mask_data)
        ids.append(file)
        
        # # calculate miou and accuracy
        # miou.append(jaccard_score(mask_data.flatten(), pred, average='binary', pos_label=1))
        # acc.append(recall_score(mask_data.flatten(), pred, average='binary', pos_label=1))


    # calculate miou and accuracy
    pred_flattened = np.concatenate([arr.flatten() for arr in predictions])
    ground_flattened = np.concatenate([arr.flatten() for arr in ground_truth])
    print('concatenated arr', pred_flattened.shape)


    # save results
    np.save(os.path.join(results_dir, 'ids.npy'), np.array(ids))
    np.save(os.path.join(results_dir, 'labels.npy'), np.array(ground_truth))
    np.save(os.path.join(results_dir, 'predictions.npy'), np.array(predictions))

    print('miou score ===>', jaccard_score(ground_flattened, pred_flattened, average='binary', pos_label=1))
    print('accuracy score ===>', recall_score(ground_flattened, pred_flattened, average='binary', pos_label=1))
    print('accuracy score ===>', classification_report(ground_flattened, pred_flattened, output_dict=True)['1']['recall'])    

# 2. Load and process TIFF files
def load_tiff_as_array(file_path):
    with rasterio.open(file_path) as src:
        return src.read().transpose(1, 2, 0)  # [Bands, H, W] -> [H, W, Bands]


# 3. Train and evaluate the model
def train_random_forest(features, labels):

    # undersample majority
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(features, labels)
    print('after undersampling', X_resampled.shape, y_resampled.shape)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_resampled, y_resampled)
    y_pred = rf.predict(X_resampled)
    print(classification_report(y_resampled, y_pred))
    return rf



def main(base_dir, results_dir, bands):

    # make results_dir
    os.makedirs(results_dir, exist_ok=True)

    train_split, test_split = merge_csv(base_dir)
    # print(len(train_split), len(test_split))

    train_data, train_labels = load_data_by_splits(base_dir, train_split, bands)

    # print(train_data.shape, train_labels.shape)

    print("============ TRAINING RF MODEL =====================")
    rf_model = train_random_forest(train_data, train_labels)

    # Save the model to a file
    dump(rf_model, os.path.join(results_dir, 'rf_model.joblib'))

    # Load the model back
    # rf_model = load(os.path.join(results_dir, 'rf_model.joblib'))

    # predict test
    print("============ RUNNING PREDICTIONS =====================")
    predict_patches(base_dir, test_split, rf_model, bands, results_dir)


if __name__ == "__main__" :

    base_dir = '/app/dev/FM4EO/data/CocoaMiningDS'
    results_dir = "/app/dev/FM4EO/results_cocoa_mining"
    bands = [2, 1,0, 7, 8, 9]

    main(base_dir, results_dir, bands)
