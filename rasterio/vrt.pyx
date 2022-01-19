"""rasterio.vrt: a module concerned with GDAL VRTs"""
import logging
import os
import random
import string
import warnings
from contextlib import ExitStack

from rasterio import gdal_version
from rasterio._err import CPLE_OpenFailedError
from rasterio._io cimport DatasetReaderBase

from rasterio.errors import NotGeoreferencedWarning

include "gdal.pxi"

import xml.etree.ElementTree as ET

import rasterio._loading
with rasterio._loading.add_gdal_dll_directories():
    import rasterio
    from rasterio._warp import WarpedVRTReaderBase, DEFAULT_NODATA_FLAG, RasterioDeprecationWarning, \
    SUPPORTED_RESAMPLING, RasterioIOError
    from rasterio.dtypes import _gdal_typename
    from rasterio.enums import MaskFlags, Resampling
    from rasterio.path import parse_path
    from rasterio.transform import TransformMethodsMixin
    from rasterio.windows import WindowMethodsMixin
    from rasterio._string import _strHighPrec


class WarpedVRT(WarpedVRTReaderBase, WindowMethodsMixin,
                TransformMethodsMixin):
    """A virtual warped dataset.

    Abstracts the details of raster warping and allows access to data
    that is reprojected when read.

    This class is backed by an in-memory GDAL VRTWarpedDataset VRT file.

    Parameters
    ----------
    src_dataset : dataset object
        The warp source.
    src_crs : CRS or str, optional
        Overrides the coordinate reference system of `src_dataset`.
    src_transfrom : Affine, optional
        Overrides the transform of `src_dataset`.
    src_nodata : float, optional
        Overrides the nodata value of `src_dataset`, which is the
        default.
    crs : CRS or str, optional
        The coordinate reference system at the end of the warp
        operation.  Default: the crs of `src_dataset`. dst_crs was
        a deprecated alias for this parameter.
    transform : Affine, optional
        The transform for the virtual dataset. Default: will be
        computed from the attributes of `src_dataset`. dst_transform
        was a deprecated alias for this parameter.
    height, width: int, optional
        The dimensions of the virtual dataset. Defaults: will be
        computed from the attributes of `src_dataset`. dst_height
        and dst_width were deprecated alias for these parameters.
    nodata : float, optional
        Nodata value for the virtual dataset. Default: the nodata
        value of `src_dataset` or 0.0. dst_nodata was a deprecated
        alias for this parameter.
    resampling : Resampling, optional
        Warp resampling algorithm. Default: `Resampling.nearest`.
    tolerance : float, optional
        The maximum error tolerance in input pixels when
        approximating the warp transformation. Default: 0.125,
        or one-eigth of a pixel.
    src_alpha : int, optional
        Index of a source band to use as an alpha band for warping.
    add_alpha : bool, optional
        Whether to add an alpha masking band to the virtual dataset.
        Default: False. This option will cause deletion of the VRT
        nodata value.
    init_dest_nodata : bool, optional
        Whether or not to initialize output to `nodata`. Default:
        True.
    warp_mem_limit : int, optional
        The warp operation's memory limit in MB. The default (0)
        means 64 MB with GDAL 2.2.
    dtype : str, optional
        The working data type for warp operation and output.
    warp_extras : dict
        GDAL extra warp options. See
        https://gdal.org/doxygen/structGDALWarpOptions.html.

    Attributes
    ----------
    src_dataset : dataset
        The dataset object to be virtually warped.
    resampling : int
        One of the values from rasterio.enums.Resampling. The default is
        `Resampling.nearest`.
    tolerance : float
        The maximum error tolerance in input pixels when approximating
        the warp transformation. The default is 0.125.
    src_nodata: int or float, optional
        The source nodata value.  Pixels with this value will not be
        used for interpolation. If not set, it will be default to the
        nodata value of the source image, if available.
    dst_nodata: int or float, optional
        The nodata value used to initialize the destination; it will
        remain in all areas not covered by the reprojected source.
        Defaults to the value of src_nodata, or 0 (gdal default).
    working_dtype : str, optional
        The working data type for warp operation and output.
    warp_extras : dict
        GDAL extra warp options. See
        https://gdal.org/doxygen/structGDALWarpOptions.html.

    Examples
    --------

    >>> with rasterio.open('tests/data/RGB.byte.tif') as src:
    ...     with WarpedVRT(src, crs='EPSG:3857') as vrt:
    ...         data = vrt.read()

    """

    def __repr__(self):
        return "<{} WarpedVRT name='{}' mode='{}'>".format(
            self.closed and 'closed' or 'open', self.name, self.mode)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if not self._closed:
            self.close()

    def __del__(self):
        if not self._closed:
            self.close()

log = logging.getLogger(__name__)

def _boundless_vrt_doc(
        src_dataset, nodata=None, background=None, hidenodata=False,
        width=None, height=None, transform=None, masked=False):
    """Make a VRT XML document.

    Parameters
    ----------
    src_dataset : Dataset
        The dataset to wrap.
    background : int or float, optional
        The background fill value for the boundless VRT.
    masked : bool
        If True, the src_dataset is replaced by its valid data mask.

    Returns
    -------
    str
        An XML text string.
    """

    nodata = nodata or src_dataset.nodata
    width = width or src_dataset.width
    height = height or src_dataset.height
    transform = transform or src_dataset.transform

    vrtdataset = ET.Element('VRTDataset')
    vrtdataset.attrib['rasterYSize'] = str(height)
    vrtdataset.attrib['rasterXSize'] = str(width)
    srs = ET.SubElement(vrtdataset, 'SRS')
    srs.text = src_dataset.crs.wkt if src_dataset.crs else ""
    geotransform = ET.SubElement(vrtdataset, 'GeoTransform')
    geotransform.text = ','.join([str(v) for v in transform.to_gdal()])

    for bidx, ci, block_shape, dtype in zip(src_dataset.indexes, src_dataset.colorinterp, src_dataset.block_shapes, src_dataset.dtypes):
        vrtrasterband = ET.SubElement(vrtdataset, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = _gdal_typename(dtype)
        vrtrasterband.attrib['band'] = str(bidx)

        if background is not None or nodata is not None:
            nodatavalue = ET.SubElement(vrtrasterband, 'NoDataValue')
            nodatavalue.text = str(background or nodata)

            if hidenodata:
                hidenodatavalue = ET.SubElement(vrtrasterband, 'HideNoDataValue')
                hidenodatavalue.text = "1"

        colorinterp = ET.SubElement(vrtrasterband, 'ColorInterp')
        colorinterp.text = ci.name.capitalize()

        complexsource = ET.SubElement(vrtrasterband, 'ComplexSource')
        sourcefilename = ET.SubElement(complexsource, 'SourceFilename')
        sourcefilename.attrib['relativeToVRT'] = "0"
        sourcefilename.attrib["shared"] = "0"
        sourcefilename.text = parse_path(src_dataset.name).as_vsi()
        sourceband = ET.SubElement(complexsource, 'SourceBand')
        sourceband.text = str(bidx)
        sourceproperties = ET.SubElement(complexsource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(width)
        sourceproperties.attrib['RasterYSize'] = str(height)
        sourceproperties.attrib['dataType'] = _gdal_typename(dtype)
        sourceproperties.attrib['BlockYSize'] = str(block_shape[0])
        sourceproperties.attrib['BlockXSize'] = str(block_shape[1])
        srcrect = ET.SubElement(complexsource, 'SrcRect')
        srcrect.attrib['xOff'] = '0'
        srcrect.attrib['yOff'] = '0'
        srcrect.attrib['xSize'] = str(src_dataset.width)
        srcrect.attrib['ySize'] = str(src_dataset.height)
        dstrect = ET.SubElement(complexsource, 'DstRect')
        dstrect.attrib['xOff'] = str((src_dataset.transform.xoff - transform.xoff) / transform.a)
        dstrect.attrib['yOff'] = str((src_dataset.transform.yoff - transform.yoff) / transform.e)
        dstrect.attrib['xSize'] = str(src_dataset.width * src_dataset.transform.a / transform.a)
        dstrect.attrib['ySize'] = str(src_dataset.height * src_dataset.transform.e / transform.e)

        if src_dataset.nodata is not None:
            nodata_elem = ET.SubElement(complexsource, 'NODATA')
            nodata_elem.text = str(src_dataset.nodata)

        if src_dataset.options is not None:
            openoptions = ET.SubElement(complexsource, 'OpenOptions')
            for ookey, oovalue in src_dataset.options.items():
                ooi = ET.SubElement(openoptions, 'OOI')
                ooi.attrib['key'] = str(ookey)
                ooi.text = str(oovalue)

        # Effectively replaces all values of the source dataset with
        # 255.  Due to GDAL optimizations, the source dataset will not
        # be read, so we get a performance improvement.
        if masked:
            scaleratio = ET.SubElement(complexsource, 'ScaleRatio')
            scaleratio.text = '0'
            scaleoffset = ET.SubElement(complexsource, 'ScaleOffset')
            scaleoffset.text = '255'

    if all(MaskFlags.per_dataset in flags for flags in src_dataset.mask_flag_enums):
        maskband = ET.SubElement(vrtdataset, 'MaskBand')
        vrtrasterband = ET.SubElement(maskband, 'VRTRasterBand')
        vrtrasterband.attrib['dataType'] = 'Byte'

        simplesource = ET.SubElement(vrtrasterband, 'SimpleSource')
        sourcefilename = ET.SubElement(simplesource, 'SourceFilename')
        sourcefilename.attrib['relativeToVRT'] = "0"
        sourcefilename.attrib["shared"] = "0"
        sourcefilename.text = parse_path(src_dataset.name).as_vsi()

        sourceband = ET.SubElement(simplesource, 'SourceBand')
        sourceband.text = 'mask,1'
        sourceproperties = ET.SubElement(simplesource, 'SourceProperties')
        sourceproperties.attrib['RasterXSize'] = str(width)
        sourceproperties.attrib['RasterYSize'] = str(height)
        sourceproperties.attrib['dataType'] = 'Byte'
        sourceproperties.attrib['BlockYSize'] = str(block_shape[0])
        sourceproperties.attrib['BlockXSize'] = str(block_shape[1])
        srcrect = ET.SubElement(simplesource, 'SrcRect')
        srcrect.attrib['xOff'] = '0'
        srcrect.attrib['yOff'] = '0'
        srcrect.attrib['xSize'] = str(src_dataset.width)
        srcrect.attrib['ySize'] = str(src_dataset.height)
        dstrect = ET.SubElement(simplesource, 'DstRect')
        dstrect.attrib['xOff'] = str((src_dataset.transform.xoff - transform.xoff) / transform.a)
        dstrect.attrib['yOff'] = str((src_dataset.transform.yoff - transform.yoff) / transform.e)
        dstrect.attrib['xSize'] = str(src_dataset.width)
        dstrect.attrib['ySize'] = str(src_dataset.height)

    return ET.tostring(vrtdataset).decode('ascii')

def form_vrt_options(resolution=None,
                     outputBounds=None,
                     targetResolution=None,
                     targetAlignedPixels=None,
                     separate=None,
                     bandList=None,
                     addAlpha=None,
                     resampleAlg=None,
                     outputSRS=None,
                     allowProjectionDifference=None,
                     srcNodata=None,
                     VRTNodata=None,
                     hideNodata=None,
                     tileIndex=None,
                     subDs=None,
                     **kwargs):
    options = []
    if resolution == 'user':
        if targetResolution is None:
            raise RuntimeError("The target resolution must be indicated when resolution is 'user'")
    else:
        if resolution is not None:
            if targetResolution is not None:
                raise RuntimeError(f"Target resolution and resolution '{resolution}' are not compatible options")
        if resolution is None:
            if targetResolution is not None:
                resolution = 'user'
            else:
                resolution = 'average'
    if targetResolution is None and targetAlignedPixels is not None:
        raise RuntimeError("Target aligned pixels must be mixed with target resolution")
    if separate is True and addAlpha is True:
        raise RuntimeError("Separate and addAlpha are not compatible options")
    if tileIndex is not None:
        options += ['-tileindex', str(tileIndex)]
    if resolution is not None:
        options += ['-resolution', str(resolution)]

    if outputBounds is not None:
        options += ['-te', _strHighPrec(outputBounds[0]), _strHighPrec(outputBounds[1]),
                    _strHighPrec(outputBounds[2]), _strHighPrec(outputBounds[3])]
    if targetResolution is not None:
        options += ['-tr', _strHighPrec(targetResolution[0]), _strHighPrec(targetResolution[1])]
    if targetAlignedPixels:
        options += ['-tap']
    if separate:
        options += ['-separate']
    if subDs:
        options += ['-sd', str(subDs)]
    if bandList is not None:
        for b in bandList:
            options += ['-b', str(b)]
    if addAlpha:
        options += ['-addalpha']
    if resampleAlg is not None:
        if resampleAlg == GDALRIOResampleAlg.GRIORA_NearestNeighbour:
            options += ['-r', 'near']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Bilinear:
            options += ['-rb']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Cubic:
            options += ['-rc']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_CubicSpline:
            options += ['-rcs']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Lanczos:
            options += ['-r', 'lanczos']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Average:
            options += ['-r', 'average']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Mode:
            options += ['-r', 'mode']
        elif resampleAlg == GDALRIOResampleAlg.GRIORA_Gauss:
            options += ['-r', 'gauss']
        else:
            options += ['-r', str(resampleAlg)]
    if outputSRS is not None:
        options += ['-a_srs', str(outputSRS)]
    if allowProjectionDifference:
        options += ['-allow_projection_difference']
    if srcNodata is not None:
        if isinstance(srcNodata, int):
            options += ['-srcnodata', str(srcNodata)]
        if isinstance(srcNodata, list) or isinstance(srcNodata, tuple):
            options += ['-srcnodata', f'{" ".join([str(nodata) for nodata in srcNodata])}']
    if VRTNodata is not None:
        if isinstance(VRTNodata, int):
            options += ['-vrtnodata', str(VRTNodata)]
        if isinstance(VRTNodata, list) or isinstance(VRTNodata, tuple):
            options += ['-vrtnodata', f'{" ".join([str(nodata) for nodata in VRTNodata])}']
    if hideNodata:
        options += ['-hidenodata']
    options_str = " ".join(options)
    options_str_enc = options_str.encode('utf-8')
    return options_str_enc


cdef GDALBuildVRTOptions* build_vrt_options(char** papszArgv, GDALBuildVRTOptionsForBinary* psOptionsForBinary) except NULL:

    cdef GDALBuildVRTOptions* vrt_options = NULL

    with nogil:
        vrt_options = GDALBuildVRTOptionsNew(papszArgv, psOptionsForBinary)
    return vrt_options

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


cdef GDALDatasetH build_vrt(const char *pszDest, int nSrcCount, GDALDatasetH *pahSrcDS, const char *const *papszSrcDSNames, const GDALBuildVRTOptions *psOptions) except NULL:
    cdef int pbUsageError = <int>0
    cdef GDALDatasetH hds_vrt = NULL
    with nogil:
        hds_vrt = GDALBuildVRT(pszDest,
                               nSrcCount,
                               pahSrcDS,
                               papszSrcDSNames,
                               psOptions,
                               &pbUsageError)
    return hds_vrt


cdef class VRTReaderBase(DatasetReaderBase):
    def __init__(self,
                 src_datasets,
                 resolution=None,
                 add_alpha=False,
                 separate=False,
                 resampling=Resampling.nearest,
                 src_nodata=DEFAULT_NODATA_FLAG,
                 vrt_nodata=DEFAULT_NODATA_FLAG,
                 allow_projection_difference=False,
                 vrt_filename=None,
                 target_aligned_pixels=None,
                 output_srs=None,
                 target_resolution=None,
                 output_bounds=None,
                 save_after_close=False,
                 bands=None,
                 tile_index=None,
                 subdataset=None,
                 hide_nodata=None):

        self.name = f"VRT {randomword(10)}"
        self.save_after_close = save_after_close
        if vrt_filename is None:
            self.vrt_filename = f"{self.name}.vrt"
        else:
            self.vrt_filename = vrt_filename

        self._src_datasets = []

        if not isinstance(src_datasets, list):
            _src_ds = None
            if isinstance(src_datasets, str):
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        _src_ds = rasterio.open(src_datasets)
                    except NotGeoreferencedWarning:
                        raise RuntimeError(f"The dataset at {src_datasets} is ungeoreferenced, which is not allowed for building VRTs.") from None
            elif isinstance(src_datasets, DatasetReaderBase):
                if src_datasets.mode != "r":
                    raise RuntimeError("The non-reading mode is not allowed") from None
                _src_ds = src_datasets
            else:
                raise RuntimeError(f"The {src_datasets} is not a valid string or rio file") from None
            self._src_datasets.append(_src_ds)
        else:
            for src_ds in src_datasets:
                _src_ds = None
                if isinstance(src_ds, str):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            _src_ds = rasterio.open(src_ds)
                        except NotGeoreferencedWarning:
                            raise RuntimeError(
                                f"The dataset at {src_ds} is ungeoreferenced, which is not allowed for building VRTs.") from None
                elif isinstance(src_ds, DatasetReaderBase):
                    if src_ds.mode != "r":
                        raise RuntimeError("The non-reading mode is not allowed") from None
                    _src_ds = src_ds
                else:
                    raise RuntimeError(f"The {src_ds} is not a valid string or rio file") from None
                self._src_datasets.append(_src_ds)

        # Guard against invalid or unsupported resampling algorithms.
        try:
            if resampling == 7:
                raise ValueError("Gauss resampling is not supported")
            Resampling(resampling)
        except ValueError:
            raise ValueError(
                "resampling must be one of: {0}".format(
                    ", ".join(['Resampling.{0}'.format(r.name) for r in SUPPORTED_RESAMPLING]))) from None

        self.mode = 'r'
        self.options = {}
        self._count = 0
        self._closed = True
        self._dtypes = []
        self._block_shapes = None
        self._nodatavals = []
        self._units = ()
        self._descriptions = ()
        self._crs = None
        self._gcps = None
        self._read = False

        if add_alpha and gdal_version().startswith('1'):
            warnings.warn("Alpha addition not supported by GDAL 1.x")
            add_alpha = False

        self.resolution = resolution
        self.resampling = Resampling.nearest
        self.src_nodata = self._src_datasets[0].nodata if src_nodata is DEFAULT_NODATA_FLAG else src_nodata
        self.vrt_nodata = self.src_nodata if vrt_nodata is DEFAULT_NODATA_FLAG else vrt_nodata
        self.output_bounds = output_bounds
        self.target_resolution = target_resolution
        self.target_aligned_pixels = target_aligned_pixels
        self.output_srs = output_srs
        self.hide_nodata = hide_nodata
        self.bands = bands
        self.tile_index = tile_index
        self.subdataset = subdataset

        cdef GDALDriverH driver = NULL
        cdef GDALBuildVRTOptions * vrt_options = NULL

        cdef GDALDatasetH* hds_list = NULL
        hds_list = <GDALDatasetH*>CPLMalloc(
            len(self._src_datasets) * sizeof(GDALDatasetH)
        )
        for i, dataset in enumerate(self._src_datasets):
            hds_ptr = (<DatasetReaderBase?> self._src_datasets[i]).handle()
            if hds_ptr == NULL:
                raise RuntimeError("Dataset is NULL")
            hds_list[i] = hds_ptr
        str_vrt_options = form_vrt_options(resolution=self.resolution,
                                           outputBounds=self.output_bounds,
                                           targetResolution=self.target_resolution,
                                           targetAlignedPixels=self.target_aligned_pixels,
                                           separate=separate,
                                           addAlpha=add_alpha,
                                           resampleAlg=self.resampling,
                                           outputSRS=self.output_srs,
                                           allowProjectionDifference=allow_projection_difference,
                                           srcNodata=self.src_nodata,
                                           VRTNodata=self.vrt_nodata,
                                           hideNodata=self.hide_nodata,
                                           tileIndex=self.tile_index,
                                           bandList=self.bands,
                                           subDs=self.subdataset)

        str_vrt_options_ptr = CSLParseCommandLine(str_vrt_options)
        if str_vrt_options_ptr == NULL:
            raise RuntimeError("String VRT options are NULL") from None
        vrt_options = build_vrt_options(str_vrt_options_ptr, NULL)

        if vrt_options == NULL:
            raise RuntimeError("VRT options are NULL") from None


        try:
            hds_vrt = build_vrt(pszDest=self.vrt_filename.encode('utf-8'),
                                nSrcCount=len(self._src_datasets),
                                pahSrcDS=hds_list,
                                papszSrcDSNames=NULL,
                                psOptions=vrt_options)
        finally:
            CSLDestroy(str_vrt_options_ptr)
            if vrt_options != NULL:
              GDALBuildVRTOptionsFree(vrt_options)
            for __src_ds in self._src_datasets:
                __src_ds.close()
        try:
            if hds_vrt == NULL:
                raise RuntimeError("VRT is NULL")
            self._hds = hds_vrt
        except CPLE_OpenFailedError as err:
            raise RasterioIOError(err.errmsg)

        self._set_attrs_from_dataset_handle()

        self._env = ExitStack()
        self._closed = False

    def read(self, indexes=None, out=None, window=None, masked=False, out_shape=None, resampling=Resampling.nearest,
                fill_value=None, out_dtype=None, **kwargs):
        return super().read(indexes=indexes, out=out, window=window, masked=masked, out_shape=out_shape,
                                    resampling=resampling, fill_value=fill_value, out_dtype=out_dtype)

    def read_masks(self, indexes=None, out=None, out_shape=None, window=None, resampling=Resampling.nearest,
                       **kwargs):
        return super().read_masks(indexes=indexes, out=out, window=window, out_shape=out_shape,
                                          resampling=resampling)




class VRT(VRTReaderBase, WindowMethodsMixin, TransformMethodsMixin):

    def __repr__(self):
        return "<{} VRT name='{}' mode='{}'>".format(
            self.closed and 'closed' or 'open', self.name, self.mode)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        if not self._closed:
            self.close()
        if not self.save_after_close:
          if os.path.exists(self.vrt_filename):
              os.remove(self.vrt_filename)

    def __del__(self):
        if not self._closed:
            self.close()
