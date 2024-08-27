import rasterio as rio
import os
import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from collections import defaultdict



def create_mining_stats(label_path='/app/dev/FM4EO/data/patch_data/2016/MASK', mining_band=0, 
                        cocoa_band=1, cocoa_threshold=0.65, image_size=128):

    stats_dict = {}

    list_images = os.listdir(label_path)

    # remove non-tif files
    list_images = [i for i in list_images if '.tif' in i]

    # area of patch
    area = image_size * image_size
    
    for fname in list_images:

        with rio.open(os.path.join(label_path, fname)) as src:
            raster_data = src.read()

            # set nodata
            raster_data = np.where(raster_data == src.nodata, 0, raster_data)

            # create mining and cocoa mask
            mining_mask = np.where(raster_data[mining_band] >1, 0, raster_data[mining_band]) 
            cocoa_mask = np.where(raster_data[cocoa_band] >= cocoa_threshold, 2, 0)

            # merge. mining area is prioritised
            mask = np.where(mining_mask !=1, cocoa_mask, mining_mask)
            mining_area = np.sum(mask == 1)
            cocoa_area = np.sum(mask == 2)

        stats_dict[fname] = [np.round(mining_area/area *100, 2), 
                             np.round(cocoa_area/area *100, 2)]

    # save dict
    with open(os.path.join(label_path,'area_stats.json'), 'w') as file:
        json.dump(stats_dict, file, indent=4)



def cluster_patches_kmeans(stats_path, num_clusters , train_percent=70):

    with open(os.path.join(stats_path,'area_stats.json'), 'r') as file:
        area_stats = json.load(file) 

    
    # Prepare lists to hold the DataFrame rows
    rows = []

    # Iterate over the dictionary items
    for key, values in area_stats.items():
        # Append each row to the list
        rows.append([key] + values)

    # Create a DataFrame from the rows list
    df = pd.DataFrame(rows, columns=['patch_name', 'mining_area', 'cocoa_area'])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['mining_area', 'cocoa_area']])


    # categorize as train or test
    def split_cluster(cluster_df):
        portion = train_percent/100.0
        train_size = int(portion * len(cluster_df))
        cluster_df = cluster_df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
        cluster_df['split'] = ['train'] * train_size + ['test'] * (len(cluster_df) - train_size)
        return cluster_df

    # Apply the function to each cluster
    df = df.groupby('cluster').apply(split_cluster).reset_index(drop=True)    

    # save df
    df.to_csv(os.path.join(stats_path,'train_test_splits.csv'), index=False)





def cluster_by_histograms(stats_path, cocoa_bins=[0, 20,40,60,80,100], 
                       mining_bins = [0,10,20,30,40,50,60,70,80,90,100], 
                       train_percent=70):
    
    with open(os.path.join(stats_path,'area_stats.json'), 'r') as file:
        area_stats = json.load(file) 
    
    # Prepare lists to hold the DataFrame rows
    rows = []
    # Iterate over the dictionary items
    for key, values in area_stats.items():
        # Append each row to the list
        rows.append([key] + values)

    # Create a DataFrame from the rows list
    df = pd.DataFrame(rows, columns=['patch_name', 'mining_area', 'cocoa_area'])

    cocoa_bin_indices = np.digitize(df['cocoa_area'], cocoa_bins) - 1
    mining_bin_indices = np.digitize(df['mining_area'], mining_bins) - 1

    # Filter valid indices (within the range of bins)
    valid_indices = (
        (cocoa_bin_indices >= 0) & (cocoa_bin_indices < len(cocoa_bins) - 0) &
        (mining_bin_indices >= 0) & (mining_bin_indices < len(mining_bins) - 0)
    )

    # Apply the filter to keep only valid indices
    cocoa_bin_indices = cocoa_bin_indices[valid_indices]
    mining_bin_indices = mining_bin_indices[valid_indices]
    indices = np.arange(len(df))[valid_indices]

    # Map samples to bins
    bin_to_samples = defaultdict(list)

    for idx, (c_bin, m_bin) in zip(indices, zip(cocoa_bin_indices, mining_bin_indices)):
        bin_to_samples[(c_bin, m_bin)].append(idx)


    # Prepare data for the DataFrame
    bin_data = []
    for bin_key, sample_indices in bin_to_samples.items():
        bin_data.append({
            'cocoa_bin': bin_key[0],
            'mining_bin': bin_key[1],
            'sample_indices': sample_indices,
            'count': len(sample_indices)
        })

    # Step 7: Create the DataFrame
    bins_df = pd.DataFrame(bin_data)    


    ## create new dataframe of train and test
    indices_list = []
    category_list = []

    # Iterate over each row in the bins_df
    for _, row in bins_df.iterrows():
        sample_indices = row['sample_indices']
        np.random.shuffle(sample_indices)  # Shuffle the indices randomly
        
        # Determine the split point for 70%-30%
        split_point = int(len(sample_indices) * train_percent/100.0)
        
        # Split into train and test
        train_indices = sample_indices[:split_point]
        test_indices = sample_indices[split_point:]
        
        # Append the indices and their categories to the lists
        indices_list.extend(train_indices)
        category_list.extend(['train'] * len(train_indices))
        indices_list.extend(test_indices)
        category_list.extend(['test'] * len(test_indices))

    # Create a new DataFrame with the results
    final_df = pd.DataFrame({
        'indices': indices_list,
        'split': category_list
    })

    mask_names = []
    for _, row in final_df.iterrows():
        idx_loc = row['indices']
        idx_name = df.iloc[idx_loc]['patch_name']
        mask_names.append(idx_name)

    final_df['patch_name'] = mask_names
    final_df.to_csv(os.path.join(stats_path,'train_test_splits.csv'), index=False)


if __name__ == "__main__":
    path = '/app/dev/FM4EO/data/patch_data/2022/MASK'
    create_mining_stats(path)
    # cluster_patches_kmeans(path, num_clusters = 5, train_percent=70)
    cluster_by_histograms(path)
    print("stats complete")