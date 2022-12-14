""" PRINCIPAL COMPONENTS ANALYSIS
--------------------------------------------------------------------------------
Script for principal components analysis (PCA) of georeferenced data.
Requires a csv file (as produced by GeoDs.datawrangle) containing the 'n'
feature values for every point i (described by feature vector Xi).
The script reads the cube, and performs principal components analysis over the
'n'-dimensions scattered data:
    - it finds the 'n' eigenvectors of the dataset;
    - it returns the eigenvectors composition as a linear compination of the
      original 'n' features, both as a csv file and a heatmap plot;
    - it returns the explained variance of the principal components in a csv
      file and the cumulative explained variance as a scree plot;
    - it reprojects the dataset to its new PC basis and returns a cube file
      similar to the input, but with transformed features;
    - it returns the 'n' georeferenced rasters of the dataset projected to each
      of the vectors of the new basis.
NOTE: *** The script does not take spatiality into consideration for
    analysis. It merely propagates (x, y) coordinates from input features
    Xi to its associated transformed features ^Xi. ***
User inputs
-----------
cube_file : str
    Path to csv file containing all data (as produced by GeoDS.datawrangle).
crs : str
    Projected coordinates reference system (preferably as EPSG).
out_folder : str
    Path to folder containing results.
columns_dict : dict
    dict of column headers in cube file to be implemented as the original
    feature basis. If left empty, all columns (but (x, y) positions!) will be
    used.
subsets : list
    List of keys in columns dict for which subsets of column to pick for the analysis
verbose : bool, default=True
    Print or not progression in the console.
Outputs
-------
pc_transformed_data : csv file (optional)
    csv of the transformed data in "out_folder/PCA_Transformed_DataFrame.csv"
eigenvectors (loadings) : csv file
    csv file of eigenvector composition in "out_folder/eigenvectors/"
eigenvectors heatmap : png figure
    Heatmap plot of eigenvector composition in "out_folder/heatmap.jpg"
scree plot: png figure
    Scree plot of cumulative explained variance in "out_folder/scree plots.jpg"
geotiffs: georeferenced tiffs
    Maps of data points projected to the new axes (PCs) "out_folder/PC*.tif"
jpg images : jpg files
    Jpg images of every geotiff file.
Notes
-----
It is recommended to redirect stdout to a file from terminal.
e.g.: python pca_v2.py > log.txt
Author(s): Ozer Ozdal 
Last update: 2022-08-24 (By Ozer Ozdal)
------------------------------------------------------------------------------
"""

import os
import socket
from datetime import datetime
from GeoDS import hypercube
#from GeoDS import eda
import eda

from GeoDS import utilities
import glob
import dask.dataframe as dd
import warnings

def main(cube_file, 
         crs, 
         out_folder, 
         columns_dict, 
         verbose=True, 
         save_df_pca=True,
         thematic_pca=True,
         explained_variance=None, 
         scaler='standard_scaler',
         no_data_value=None
         ):
    if verbose:
        t0 = datetime.now()
        print('-----------------------------------------------------------')
        print('Starting new job at ' + str(t0) + '.')
        print('-----------------------------------------------------------\n')

    if verbose: print(f'Running on {socket.gethostname()}')
    if verbose: print(f'Running from {os.getcwd()}\n')
    # Make subfolders
    os.makedirs(out_folder, exist_ok=True)
    #os.makedirs(out_folder + '/eigenvectors', exist_ok=True)
    #os.makedirs(out_folder + '/geotiffs', exist_ok=True)

    # Read file
    if verbose: print(f'Reading {cube_file}...')
    cube = hypercube.HyperCube(input_data=cube_file, x_field='x', y_field='y', crs=crs, columns_dict=columns_dict)    
    if type(cube.df) == dd.core.DataFrame: 
        if verbose: print('\nComputing Dask DataFrame...')
        cube.df = cube.df.compute()
    else:
        pass
    
    # Do PCA (Thematic or Normal PCA depending on the thematic_pca flag)       
    if verbose:
        if thematic_pca: 
            print('\nPreparing and performing Thematic PCA...')
        else:
            print('\nPreparing and performing PCA...')        
     
    df_pca, thematic_pca_objects = eda.pca_dataframe(cube.df, 
              explained_variance=explained_variance,
              scaler=scaler,
              random_state=42,
              coordinates_index=None, 
              output_folder=out_folder,
              thematic_pca=thematic_pca,
              columns=columns_dict,
              save_df_pca=save_df_pca,
              no_data_value=no_data_value)        

    if verbose:
        if thematic_pca: 
            print("Thematic PCA Done.")
        else:
            print("PCA Done.")

    if verbose: print('\nConverting the PCs to Geotiffs...')
    # Convert the PCs to Geotiffs
    X_coord = ['x', 'X'] # Possible x coordinates
    Y_coord = ['y', 'Y'] # Possible y coordinates
    Z_coord = ['z', 'Z'] # Possible z coordinates

    X_coordinates = df_pca.columns.intersection(X_coord).tolist() # X coordinates in the dataframe
    Y_coordinates = df_pca.columns.intersection(Y_coord).tolist() # Y coordinates in the dataframe
    Z_coordinates = df_pca.columns.intersection(Z_coord).tolist() # Z coordinates in the dataframe

    coordinates = X_coordinates + Y_coordinates + Z_coordinates # All coordinates in the dataframe
        
    if len(coordinates) == 2 and len(X_coordinates) == 1 and len(Y_coordinates) == 1:
        utilities.csv_to_raster(input_csv=df_pca, 
                    output_directory=os.path.join(out_folder), 
                    columns=df_pca.columns.difference(coordinates).tolist(), 
                    categories=None, 
                    x_field=X_coordinates[0],
                    y_field=Y_coordinates[0], 
                    dstCRS=crs)            
    elif len(coordinates) == 3 and len(X_coordinates) == 1 and len(Y_coordinates) == 1 and len(Z_coordinates) == 1:
        warnings.warn("The utilities.csv_to_raster function does not work for three dimensions." + os.linesep +
                      "Update the function in three dimensions" + os.linesep +
                      "or update this elif block by writing another function that does a three-dimensional visualization")
    else:
        warnings.warn("There is a problem in the coordinates of the dataframe. Please check your dataframe.")
            
    # Lets convert them to jpg as well to include them in our evaluation template
    if verbose: print("Converting each raster to .jpg...")
    tifs = glob.glob(os.path.join(out_folder, '*.tif'))        
            
    for t in tifs:
        filename, extension, directory = utilities.Path_Info(t)
        utilities.geotiff_to_jpg(t, os.path.join(out_folder, filename + '.jpg'))
        
    if verbose: print(f'Generating a scree plot in {out_folder}')
    for pca_objects in thematic_pca_objects:
        # Generating as many scree plots as the number of Thematic PCA domains!
        if verbose: print(f'Generating a scree plot for {str(pca_objects)}')
        fig, ax = eda.scree_plot(thematic_pca_objects[pca_objects], out_folder, title = 'scree_plot_'+str(pca_objects))   

    if verbose: print(f'Generating PC loadings in {out_folder}')
    for pca_objects in thematic_pca_objects:
        eda.plot_pc_loadings(pca_transformer=thematic_pca_objects[pca_objects],
                             feature_list=thematic_pca_objects[pca_objects].feature_names_in_.tolist(),
                             output_directory=out_folder, 
                             dpi=150,
                             thematic_domain = str(pca_objects))

    if verbose: print(f'Generating eigenvectors heatmap and csv in {out_folder}')
    for pca_objects in thematic_pca_objects:
        eda.plot_loadings_heatmap(pca_transformer=thematic_pca_objects[pca_objects],
                             original_labels=thematic_pca_objects[pca_objects].feature_names_in_.tolist(),
                             output_directory=out_folder,
                             title='heatmap_'+str(pca_objects))      
        
    if verbose: print("Generating the evaluation template")
    for pca_objects in thematic_pca_objects:
        eda.pc_evaluation_template(thematic_pca_objects[pca_objects], 
                             thematic_pca_objects[pca_objects].feature_names_in_.tolist(),
                             output_directory = out_folder, 
                             jpgs_folder=out_folder,
                             thematic_domain = str(pca_objects))        
            
    if verbose:
        t1 = datetime.now()
        print('\nJob ended at ' + str(t1) + '.')

# ------------------------------------------------------------------------------

# -------- Script run ----------------------------------------------------------
# ------------------------------------------------------------------------------
# Example Below
# ------------------------------------------------------------------------------
#if __name__ == "__main__":
#    main(_PATH_TO_THE_CUBE, 
#         crs, 
#         test_folder, 
#         _PATH_TO_THE_JSON,  
#         verbose=True, 
#         save_df_pca=True,
#         thematic_pca=True,
#         explained_variance=None, 
#         scaler='standard_scaler')

# ------------------------------------------------------------------------------