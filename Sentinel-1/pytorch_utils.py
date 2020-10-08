
import json
import os
import numpy as np


# original labels
LABELS = [
    'Continuous urban fabric',
    'Discontinuous urban fabric',
    'Industrial or commercial units',
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Non-irrigated arable land',
    'Permanently irrigated land',
    'Rice fields',
    'Vineyards',
    'Fruit trees and berry plantations',
    'Olive groves',
    'Pastures',
    'Annual crops associated with permanent crops',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland',
    'Moors and heathland',
    'Sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Sparsely vegetated areas',
    'Burnt areas',
    'Inland marshes',
    'Peatbogs',
    'Salt marshes',
    'Salines',
    'Intertidal flats',
    'Water courses',
    'Water bodies',
    'Coastal lagoons',
    'Estuaries',
    'Sea and ocean'
]
# the new labels
NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]
# removed labels from the original 43 labels
REMOVED_LABELS = [
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Bare rock',
    'Burnt areas',
    'Intertidal flats'
]
# merged labels
GROUP_LABELS = {
    'Continuous urban fabric':'Urban fabric',
    'Discontinuous urban fabric':'Urban fabric',
    'Non-irrigated arable land':'Arable land',
    'Permanently irrigated land':'Arable land',
    'Rice fields':'Arable land',
    'Vineyards':'Permanent crops',
    'Fruit trees and berry plantations':'Permanent crops',
    'Olive groves':'Permanent crops',
    'Annual crops associated with permanent crops':'Permanent crops',
    'Natural grassland':'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas':'Natural grassland and sparsely vegetated areas',
    'Moors and heathland':'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation':'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes':'Inland wetlands',
    'Peatbogs':'Inland wetlands',
    'Salt marshes':'Coastal wetlands',
    'Salines':'Coastal wetlands',
    'Water bodies':'Inland waters',
    'Water courses':'Inland waters',
    'Coastal lagoons':'Marine waters',
    'Estuaries':'Marine waters',
    'Sea and ocean':'Marine waters'
}

#Country Serbia Labels
LABELS_SERBIA = [
    'Continuous urban fabric',
    'Discontinuous urban fabric',
    'Industrial or commercial units',
    'Road and rail networks and associated land',
    'Port areas',
    'Airports',
    'Mineral extraction sites',
    'Dump sites',
    'Construction sites',
    'Green urban areas',
    'Sport and leisure facilities',
    'Non-irrigated arable land',
    'Vineyards',
    'Fruit trees and berry plantations',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland',
    'Moors and heathland',
    'Sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Sparsely vegetated areas',
    'Burnt areas',
    'Inland marshes',
    'Water courses',
    'Water bodies',
    'Broad-leaved forest'
]

LABELS_NOT_IN_SERBIA = [
    'Peatbogs',
    'Salt marshes',
    'Salines',
    'Intertidal flats',
    'Coastal lagoons',
    'Estuaries',
    'Sea and ocean',
    'Agro-forestry areas',
    'Annual crops associated with permanent crops',
    'Olive groves',
    'Rice fields',
    'Permanently irrigated land'
]



def cls2multiHot_old(cls_vec,labels):
    """ 
    create old multi hot label
    """
    tmp = np.zeros((len(labels),))
    for cls_nm in cls_vec:
        tmp[labels.index(cls_nm)] = 1

    return tmp

def read_scale_raster(file_path, GDAL_EXISTED, RASTERIO_EXISTED):
    """
    read raster file with specified scale
    :param file_path:
    :param scale:
    :return:
    """
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio

    if GDAL_EXISTED:
        band_ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = raster_band.ReadAsArray()

    elif RASTERIO_EXISTED:
        band_ds = rasterio.open(file_path)
        band_data = np.array(band_ds.read(1))
    
    return band_data

def parse_json_labels(f_j_path):
    """
    parse meta-data json file for big earth to get image labels
    :param f_j_path: json file path
    :return:
    """
    with open(f_j_path, 'r') as f_j:
        j_f_c = json.load(f_j)
    return j_f_c['labels']


class dataGenBigEarthTiff:
    def __init__(self, sentinel1Dir=None,
                bigEarthDir=None,
                patch_names_list=None,
                RASTERIO_EXISTED=None, GDAL_EXISTED=None
                ):

        self.sentinel1Dir = sentinel1Dir
        self.bigEarthDir = bigEarthDir
        
        
        self.GDAL_EXISTED = GDAL_EXISTED
        self.RASTERIO_EXISTED = RASTERIO_EXISTED

        self.total_patch = patch_names_list[0] + patch_names_list[1] + patch_names_list[2]

    def __len__(self):

        return len(self.total_patch)
    
    def __getitem__(self, index):

        return self.__data_generation(index)

    def __data_generation(self, idx):

        imgNmS1 = self.total_patch[idx]
        imgNmS2 = s1NameToS2(imgNmS1)
        


        polarVHs_array = []
        polarVVs_array = []


        polarVHs_array.append(read_scale_raster(os.path.join(self.sentinel1Dir, imgNmS1, imgNmS1+'_VH'+'.tif'), self.GDAL_EXISTED, self.RASTERIO_EXISTED))                    
        polarVVs_array.append(read_scale_raster(os.path.join(self.sentinel1Dir, imgNmS1, imgNmS1+'_VV'+'.tif'), self.GDAL_EXISTED, self.RASTERIO_EXISTED))      

        polarVHs_array = np.asarray(polarVHs_array).astype(np.float32)
        polarVVs_array = np.asarray(polarVVs_array).astype(np.float32)


        labels = parse_json_labels(os.path.join(self.bigEarthDir, imgNmS2, imgNmS2+'_labels_metadata.json'))
        oldMultiHots = cls2multiHot_old(labels,LABELS_SERBIA)
        oldMultiHots.astype(int)

        sample = {'polarVHs': polarVHs_array, 'polarVVs': polarVVs_array, 
                'patch_name': imgNmS1, 'multi_hots_o':oldMultiHots}
               
        return sample
    
    
def s1NameToS2(s1Name):
    s2Name = s1Name.replace('S1_','')
    return s2Name
    
    
    

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    import pyarrow as pa
    
    return pa.serialize(obj).to_buffer()


def prep_lmdb_files(root_folder, sentinel1Directory, out_folder, patch_names_list, GDAL_EXISTED, RASTERIO_EXISTED,name):
    
    from torch.utils.data import DataLoader
    import lmdb

    dataGen = dataGenBigEarthTiff(
                                sentinel1Dir = sentinel1Directory,
                                bigEarthDir = root_folder,
                                patch_names_list=patch_names_list,
                                GDAL_EXISTED=GDAL_EXISTED,
                                RASTERIO_EXISTED=RASTERIO_EXISTED
                                )

    nSamples = len(dataGen)
    map_size_ = (dataGen[0]['polarVHs'].nbytes + dataGen[0]['polarVVs'].nbytes )*10*len(dataGen)
    data_loader = DataLoader(dataGen, num_workers=4, collate_fn=lambda x: x)

    db = lmdb.open(os.path.join(out_folder, name), map_size=map_size_)

    txn = db.begin(write=True)
    patch_names = []
    for idx, data in enumerate(data_loader):
        polarVH, polarVV, patch_name, multiHots_o = data[0]['polarVHs'], data[0]['polarVVs'], data[0]['patch_name'], data[0]['multi_hots_o']
        txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pyarrow((polarVH, polarVV, multiHots_o)))
        patch_names.append(patch_name)

        if idx % 10000 == 0:
            print("[%d/%d]" % (idx, nSamples))
            txn.commit()
            txn = db.begin(write=True)
    
    txn.commit()
    keys = [u'{}'.format(patch_name).encode('ascii') for patch_name in patch_names]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


