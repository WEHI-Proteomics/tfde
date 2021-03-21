# -*- coding: utf-8 -*-
"""Python wrapper for timsdata.dll"""

import numpy as np
import sqlite3
import os, sys
from ctypes import *

if sys.platform[:5] == "win32":
    libname = "timsdata.dll"
elif sys.platform[:5] == "linux":
    if os.path.exists('/home/ubuntu/'):
        libname = "/home/ubuntu/otf-peak-detect/timstof-sdk/libtimsdata.so"
    else:
        libname = "/home/daryl/otf-peak-detect/timstof-sdk/libtimsdata.so"
else:
    raise Exception("Unsupported platform.")

dll = cdll.LoadLibrary(libname)
dll.tims_open.argtypes = [ c_char_p, c_uint32 ]
dll.tims_open.restype = c_uint64
dll.tims_close.argtypes = [ c_uint64 ]
dll.tims_close.restype = None
dll.tims_get_last_error_string.argtypes = [ c_char_p, c_uint32 ]
dll.tims_get_last_error_string.restype = c_uint32
dll.tims_has_recalibrated_state.argtypes = [ c_uint64 ]
dll.tims_has_recalibrated_state.restype = c_uint32
dll.tims_read_scans_v2.argtypes = [ c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32 ]
dll.tims_read_scans_v2.restype = c_uint32

convfunc_argtypes = [ c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32 ]

dll.tims_index_to_mz.argtypes = convfunc_argtypes
dll.tims_index_to_mz.restype = c_uint32
dll.tims_mz_to_index.argtypes = convfunc_argtypes
dll.tims_mz_to_index.restype = c_uint32

dll.tims_scannum_to_oneoverk0.argtypes = convfunc_argtypes
dll.tims_scannum_to_oneoverk0.restype = c_uint32
dll.tims_oneoverk0_to_scannum.argtypes = convfunc_argtypes
dll.tims_oneoverk0_to_scannum.restype = c_uint32

dll.tims_scannum_to_voltage.argtypes = convfunc_argtypes
dll.tims_scannum_to_voltage.restype = c_uint32
dll.tims_voltage_to_scannum.argtypes = convfunc_argtypes
dll.tims_voltage_to_scannum.restype = c_uint32

def throwLastTimsDataError (dll_handle):
    """Throw last TimsData error string as an exception."""

    len = dll_handle.tims_get_last_error_string(None, 0)
    buf = create_string_buffer(len)
    dll_handle.tims_get_last_error_string(buf, len)
    raise RuntimeError(buf.value)

# Decodes a properties BLOB of type 12 (array of strings = concatenation of
# zero-terminated UTF-8 strings). (The BLOB object returned by an SQLite query can be
# directly put into this function.) \returns a list of unicode strings.
def decodeArrayOfStrings (blob):
    if blob is None:
        return None # property not set

    if len(blob) == 0:
        return [] # empty list

    blob = bytearray(blob)
    if blob[-1] != 0:
        raise ValueError("Illegal BLOB contents.") # trailing nonsense

    if sys.version_info.major == 2:
        return unicode(str(blob), 'utf-8').split('\0')[:-1]
    if sys.version_info.major == 3:
        return str(blob, 'utf-8').split('\0')[:-1]

class TimsData:

    def __init__ (self, analysis_directory, use_recalibrated_state=False):

        if sys.version_info.major == 2:
            if not isinstance(analysis_directory, unicode):
                raise ValueError("analysis_directory must be a Unicode string.")
        if sys.version_info.major == 3:
            if not isinstance(analysis_directory, str):
                raise ValueError("analysis_directory must be a string.")

        self.dll = dll

        self.handle = self.dll.tims_open(
            analysis_directory.encode('utf-8'),
            1 if use_recalibrated_state else 0 )
        if self.handle == 0:
            throwLastTimsDataError(self.dll)

        self.conn = sqlite3.connect(os.path.join(analysis_directory, "analysis.tdf"))

        self.initial_frame_buffer_size = 128 # may grow in readScans()

    def __del__ (self):
        if hasattr(self, 'handle'):
            self.dll.tims_close(self.handle)

    def __callConversionFunc (self, frame_id, input_data, func):

        if type(input_data) is np.ndarray and input_data.dtype == np.float64:
            # already "native" format understood by DLL -> avoid extra copy
            in_array = input_data
        else:
            # convert data to format understood by DLL:
            in_array = np.array(input_data, dtype=np.float64)

        cnt = len(in_array)
        out = np.empty(shape=cnt, dtype=np.float64)
        success = func(self.handle, frame_id,
                       in_array.ctypes.data_as(POINTER(c_double)),
                       out.ctypes.data_as(POINTER(c_double)),
                       cnt)

        if success == 0:
            throwLastTimsDataError(self.dll)

        return out

    def indexToMz (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_index_to_mz)

    def mzToIndex (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_mz_to_index)

    def scanNumToOneOverK0 (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_scannum_to_oneoverk0)

    def oneOverK0ToScanNum (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_oneoverk0_to_scannum)

    def scanNumToVoltage (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_scannum_to_voltage)

    def voltageToScanNum (self, frame_id, mzs):
        return self.__callConversionFunc(frame_id, mzs, self.dll.tims_voltage_to_scannum)


    # Output: list of tuples (indices, intensities)
    def readScans (self, frame_id, scan_begin, scan_end):

        # buffer-growing loop
        while True:
            cnt = int(self.initial_frame_buffer_size) # necessary cast to run with python 3.5
            buf = np.empty(shape=cnt, dtype=np.uint32)
            len = 4 * cnt

            required_len = self.dll.tims_read_scans_v2(self.handle, frame_id, scan_begin, scan_end,
                                                    buf.ctypes.data_as(POINTER(c_uint32)),
                                                    len)
            if required_len == 0:
                throwLastTimsDataError(self.dll)

            if required_len > len:
                if required_len > 16777216:
                    # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.initial_frame_buffer_size = required_len / 4 + 1 # grow buffer
            else:
                break

        result = []
        d = scan_end - scan_begin
        for i in range(scan_begin, scan_end):
            npeaks = buf[i-scan_begin]
            indices     = buf[d : d+npeaks]
            d += npeaks
            intensities = buf[d : d+npeaks]
            d += npeaks
            result.append((indices,intensities))

        return result
