# Generation of Training/Test/Validation Splits
This part is based on https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models.

After dowloading the Sentinel-1 and Sentinel-2 images, they need to be prepared to use in train,validation and test steps. `prep_splits.py` in Sentinel-1 and Sentinel-2 folders are responsible to generate consumable data files(LMDB files) whcih are suitable to use with PyTorch. Suggested splits for Sentinel-1 and Sentinel-2 Patches can be found under `splits` folders. 

Arguments for `prep_splits.py` in **Sentinel-1** folder:
* `-r` or `--root_folder`: The root folder containing the Sentinel-2 images you have previously downloaded. Although this script runs to create LMDB files of Sentinel-1 images, Sentinel-2 image pairs of Sentinel-1 Images are need in order to get labels of the images.
* `s1` or `--s1_root_folder`: The root folder containg the Sentinel-1 images you have previously downloaded. 
* `-o` or `--out_folder`: The folder path will containing resulting LMDB files
* `-n` or `--splits`: CSV files each of which contain list of Sentinel-1 patch names
* `-name`: The name of the folder which will have resulting files

Arguments for `prep_splits.py` in **Sentinel-2** folder:
* `-r` or `--root_folder`: The root folder containing the Sentinel-2 images you have previously downloaded.
* `-o` or `--out_folder`: The folder path will containing resulting LMDB files
* `-n` or `--splits`: CSV files each of which contain list of Sentinel-1 patch names
* `-name`: The name of the folder which will have resulting files
* `--serbia`: Serbia patches does not have all classes which are represented in BigEarthNet. In order to have a correct multi hot encoding during processing of labels, this argument should be set as True while Sentinel-2 Serbia patches have been used in the script. 


To run the script, either the GDAL or the rasterio package should be installed. The PyTorch package should also be installed. The script is tested with Python 3.6.7, PyTorch 1.2.0, and CentOS Linux 7 (TU Berlin High Performance Cluster) . 

