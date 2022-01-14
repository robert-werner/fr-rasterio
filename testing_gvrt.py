from matplotlib import pyplot
from rasterio.vrt import VRT

import os
import rasterio

tifs = []
ic = "/home/leonid/Desktop/rio-vrt-test"
for _ in os.listdir(ic):
    if '.tif' in _:
        tifs.append(rasterio.open(os.path.join(ic, _)))
with VRT(src_datasets=tifs, src_nodata=0, vrt_nodata=0, add_alpha=True) as vrt:
    c1 = vrt.read(1)
    pyplot.imshow(c1)
    pyplot.show()
