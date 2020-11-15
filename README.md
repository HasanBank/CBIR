This project has been created in the scope of my master thesis in computer science, TU Berlin.

# Generation of Training/Test/Validation Splits
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

# Training
* `--S1LMDBPth` : The folder path contains Sentinel-1 LMDB dataset previously created.
* `--S2LMDBPth` : The folder path contains Sentinel-2 LMDB dataset previously created.
* `-b` or `--batch-size` : Mini-batch size
* `--epochs` : Total epoch number
* `--k` : number of retrived images per query. Default 20.
* `--lr`: initial learning rate. Default 0.001
* `--num_workers` : number of workers for data loading in pytorch. Default 8.
* `--bits` : hash length. Default 16.
* `--serbia` : It should be set as True when Serbia patches are used. 
* `--train_csvS1`: Path of the CSV file which shows Sentinel 1 Train Patches
* `--val_csvS1`: Path of the CSV file which shows Sentinel 1 Validation Patches
* `--test_csvS1`: Path of the CSV file which shows Sentinel 1 Test Patches
* `-loss` or `--lossFunction` : Two loss function has been implemented. These are: 'MSELoss' and 'TripletLoss'.



# Testing
* `--S1LMDBPth` : The folder path contains Sentinel-1 LMDB dataset previously created.
* `--S2LMDBPth` : The folder path contains Sentinel-2 LMDB dataset previously created.
* `--S1Dir` : The folder path contains raw Sentinel-1 patches
* `--S2Dir` : The folder path contains raw Sentinel-2 patches
* `-b` or `--batch-size` : Mini-batch size
* `--bits` : hash length. Default 16.
* `--checkpoint_pth` : path to the pretrained weights file which is from train script.
* `--num_workers` : number of workers for data loading in pytorch. Default 8.
* `--train_csvS1`: Path of the CSV file which shows Sentinel 1 Train Patches
* `--val_csvS1`: Path of the CSV file which shows Sentinel 1 Validation Patches
* `--test_csvS1`: Path of the CSV file which shows Sentinel 1 Test Patches
* `--dataset`: Path of the hashed data.
* `--k` : number of retrived images per query. Default 20.
* `--serbia` : It should be set as True when Serbia patches are used. 
















