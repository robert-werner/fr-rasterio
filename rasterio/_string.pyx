import logging

include "gdal.pxi"

log = logging.getLogger(__name__)

def _is_str_or_unicode(o):
    return isinstance(o, (str, type('')))

def _strHighPrec(x):
    return x if _is_str_or_unicode(x) else '%.18g' % x