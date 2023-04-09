"""
To read .nex or .nex5 files, use the following code:
    import nex
    reader = nex.Reader()
    fileData = reader.ReadNexFile('C:\\Data\\file.nex')
    fileData1 = reader.ReadNexFile('C:\\Data\\file.nex5')
    print(fileData['Variables'][0])

If your files are larger than a few MB, use numpy version of the reader:
    import nex
    reader = nex.Reader(useNumpy=True)
    fileData = reader.ReadNexFile('C:\\Data\\LargeFile.nex')

See comments below for the description of the content of fileData.

To write .nex file, use this code:
    timestampFrequency = 50000
    writer = nex.NexWriter(timestampFrequency)
then, add variable data using Add... methods in NexWriter class
(see method doc strings below for more info):
    writer.AddContVarWithSingleFragment('cont1', 0, 10000, [5, 6, 7, 8])
    writer.AddContVarWithSingleFragment('cont2', 0, 10000, [9, 10, 11, 12])
then, use WriteNexFile method:
    writer.WriteNexFile('C:\\Data\\python.nex')

If your files are larger than a few MB, use numpy version of the NexWriter:
    import nex
    import numpy as np
    timestampFrequency = 50000
    writer = nex.NexWriter(timestampFrequency, useNumpy=True)
    writer.AddNeuron('neuron1', np.array([1, 2, 3, 4]))
    writer.AddContVarWithSingleFragment('cont1', 2, 10000, np.array([5, 6, 7, 8]))
    writer.WriteNex5File('C:\\Data\\pythonWithFloatContValues.nex5', saveContValuesAsFloats=1)

fileData:
    fileData["FileHeader"] -- file header:
        fileData["FileHeader"]["Comment"] -- comment
        fileData["FileHeader"]["Beg"] -- file data start in ticks
        fileData["FileHeader"]["End"] -- file data end in ticks
        fileData["FileHeader"]["NexFileVersion"] -- file version
        fileData["FileHeader"]["MagicNumber"] -- magic number
        fileData["FileHeader"]["Frequency"] -- timestamp frequency
        fileData["FileHeader"]["MetaOffset"] -- location of metadata (position in file)
        fileData["FileHeader"]["NumVars"] -- number of variables

    fileData["MetaData"] -- file metadata: various extra information fields that do not fit
        into file or variable headers

    fileData["Variables"] -- a list of data variables
        For each variable var in fileData["Variables"]:
            var["Timestamps"] -- array of timestamps in seconds (neurons, events, waveforms)
            var["Intervals"] -- array of intervals in seconds (interval variables)
            var["WaveformValues"] -- array of waveform values in milliVolts (waveform variables)
            
            for marker variables:
                var["MarkerFieldNames"] -- array of marker field names
                var["Markers"] -- array of marker values for all fields
                var["Fields"] -- array of marker fields
                for each field:
                    var["Fields"][i]["Name"] -- the name of the marker field
                    var["Fields"][i]["Marker"] -- marker values
            
            for continuous variables:
                var["FragmentCounts"] -- array of fragment counts (how many samples in each fragment)
                var["FragmentIndexes"] -- array of fragment indexes (index of the first data point in each fragment)
                var["ContinuousValues"] -- array of continuous values in milliVolts
                var["FragmentTimestamps"] -- array of fragment timestamps (timestamp of the first data point in each fragment)
            var["Header"] -- variable header:
                var["Header"]["Type"] -- variable type: 0 - neuron, 1 - event, 2- interval, 3 - waveform, 4 - pop. vector, 5 - continuously recorded, 6 - marker
                var["Header"]["Name"] -- variable name
                var["Header"]["Version"] -- variable version in file
                var["Header"]["DataOffset"] -- where the data array for this variable is located in the file
                var["Header"]["Count"] --  neuron variable: number of timestamps
                             event variable: number of timestamps
                             interval variable: number of intervals
                             waveform variable: number of waveforms
                             continuous variable: number of fragments
                             population vector: number of weights
                var["Header"]["TsDataType"] -- if 0, timestamps are stored as 32-bit integers; 
                            if 1, timestamps are stored as 64-bit integers;  supported by NeuroExplorer version 5.100 or greater
                var["Header"]["ContDataType"] -- waveforms and continuous variables only, 
                            if 0, waveform and continuous values are stored as 16-bit integers; 
                            if 1, waveform and continuous values are stored as 32-bit floating point values in units specified in Units field
                var["Header"]["SamplingRate"] -- waveforms and continuous variables only, waveform or continuous variable sampling frequency in Hertz
                var["Header"]["ADtoMV"] -- waveforms and continuous variables only, coefficient 
                            to convert from A/D values stored in file to units. 
                            A/D values in fileData are already scaled to units
                            see formula below UnitsOffset below; ignored if ContinuousDataType == 1 
                var["Header"]["MVOffset"] -- waveforms and continuous variables only, 
                            this offset is used to convert A/D values stored in file to units:
                            value_in_units = raw * ADtoUnitsCoefficient + UnitsOffset; ignored if ContinuousDataType == 1
                             A/D values in fileData are already scaled to units.
                var["Header"]["NPointsWave"] -- waveform variable: number of data points in each wave
                                           continuous variable: overall number of data points in the variable
                var["Header"]["PreThrTime"] -- waveform variables only, pre-threshold time in seconds
                            if waveform timestamp in seconds is t, 
                            then the timestamp of the first point of waveform is t - PrethresholdTimeInSeconds
                var["Header"]["MarkerDataType"] -- marker events only,
                            if 0, marker values are stored as strings; 
                            if 1, marker values are stored as 32-bit unsigned integers
                var["Header"]["MarkerLength"] -- marker events only, how many characters are in each marker value; ignored if MarkerDataType is 1
                var["Header"]["ContFragIndexType"] -- continuous variables only, 
                            if 0, indexes of first data point in fragments are stored as unsigned 32-bit integers; 
                            if 1, indexes of first data point in fragments are stored as unsigned 64-bit integers; not supported in NeuroExplorer as of October 2017 (version 5.107)
                var["Header"]["Units"] -- waveforms and continuous variables only, units that should be used for the variable values
                            not supported as of January 2022 (version 5.312)
                var["Header"]["NMarkers"] -- marker events only, how many values are associated with each marker

"""

import sys
import os
import struct
import json
import numbers


class NexFileVarType:
    """
    Constants for .nex and .nex5 variable types
    """
    NEURON = 0
    EVENT = 1
    INTERVAL = 2
    WAVEFORM = 3
    POPULATION_VECTOR = 4
    CONTINUOUS = 5
    MARKER = 6
    
class DataFormat:
    # constants for reading data
    INT16 = 0
    UINT16 = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    FLOAT32 = 6
    FLOAT64 = 7
    
    @staticmethod
    def NumBytesPerItem(dataType):
        numBytes = [2,2,4,4,8,8,4,8]
        return numBytes[dataType]
    
    @staticmethod
    def StructTypeFromDataType(dataType):
        formats = ['h', 'H', 'i', 'I', 'q', 'Q', 'f', 'd']
        return formats[dataType]    


class Reader(object):
    """
    Nex file reader class.
    """
    def __init__(self, useNumpy=True):
        """
        Constructor
        :param useNumpy: option to use numpy to read data arrays.
        """
        self.theFile = None
        self.fileData = None
        self.useNumpy = useNumpy
        self.fromTicksToSeconds = 1
        
    def ReadNex5File(self, filePath:str) -> dict:
        """ 
        Reads data from .nex5 file.
        :param filePath: full path of file
        :return: file data
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == '.nex':
            return self.ReadNexFile(filePath)
        self.fileData = {}
        self.theFile = open(filePath, 'rb')

        # read file header
        self.fileData['FileHeader'] = self._ReadNex5FileHeader()
        self.fileData['Variables'] = []

        # read variable headers and create variables
        for varNum in range(self.fileData['FileHeader']['NumVars']):
            var = {'Header': self._ReadNex5VarHeader()}
            self.fileData['Variables'].append(var)

        # read variable data
        self._ReadData()

        # read metadata
        metaOffset = self.fileData['FileHeader']['MetaOffset']
        if metaOffset > 0:
            self.theFile.seek(0, os.SEEK_END)
            size = self.theFile.tell()
            if metaOffset < size:
                self.theFile.seek(metaOffset)
                metaString = self.theFile.read(size - metaOffset).decode('utf-8').strip('\x00')
                metaString = metaString.strip()
                try:
                    self.fileData['MetaData'] = json.loads(metaString)
                except Exception as error:
                    print('Invalid file metadata: ' + repr(error))

        self.theFile.close()
        return self.fileData

    def ReadNexFile(self, filePath:str) -> dict:
        """
        Reads data from .nex file.
        :param filePath:
        :return: file data
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == '.nex5':
            return self.ReadNex5File(filePath)

        self.fileData = {}
        self.theFile = open(filePath, 'rb')

        self.fileData['FileHeader'] = self._ReadFileHeader()
        self.fileData['Variables'] = []

        for varNum in range(self.fileData['FileHeader']['NumVars']):
            var = {'Header': self._ReadVarHeader()}
            self.fileData['Variables'].append(var)

        self._ReadData()

        self.theFile.close()
        return self.fileData

    def _ReadData(self):
        for var in self.fileData['Variables']:
            self.theFile.seek(var['Header']['DataOffset'])
            varType = var['Header']['Type']
            methods = [self._ReadTimestamps, self._ReadTimestamps, self._ReadIntervals, 
                       self._ReadWaveforms, self._ReadPopVectors, self._ReadContinuous, self._ReadMarker ]
            methods[varType](var)

    
    def _ToString(self, theBytes, discardAfterFirstZero=False):
        """Converts bytes read from a file to a string. 
           Some .nex file writers may write garbage after the zero-terminaged string.
           We need to discard this garbage before converting bytes to string.
        """
        if discardAfterFirstZero:
            theBytesBeforeZero = theBytes.split(b'\0', 1)[0]
            try:
                return theBytesBeforeZero.decode('ascii')
            except:
                return theBytesBeforeZero.decode('ascii', errors='replace')
                
        try:
            str = theBytes.decode('utf-8').strip('\x00')
            return str
        except:
            # try to discard garbage after zero and then decode
            theBytesBeforeZero = theBytes.split(b'\0', 1)[0]
            try:
                return theBytesBeforeZero.decode('utf-8').strip('\x00')
            except:
                return theBytesBeforeZero.decode('utf-8', errors='replace').strip('\x00')
                

    def _ReadNex5FileHeader(self):
        fileHeaderFormat = '<i i 256s d q i Q q 56s'
        headerBytes = self.theFile.read(struct.calcsize(fileHeaderFormat))
        if len(headerBytes) < struct.calcsize(fileHeaderFormat):
            raise ValueError("unable to read file header")
        fhValues = struct.unpack(fileHeaderFormat, headerBytes)
        keys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'NumVars', 'MetaOffset', 'End', 'Padding']
        fileHeader = dict(zip(keys, fhValues))
        del fileHeader['Padding']

        if fileHeader['MagicNumber'] != 894977358 or fileHeader['Frequency'] <= 0:
            raise ValueError('Invalid .nex5 file')

        fileHeader['Comment'] = self._ToString(fileHeader['Comment'])
        self.tsFreq = fileHeader['Frequency']
        self.fromTicksToSeconds = 1.0 / self.tsFreq
        fileHeader['Beg'] /= self.tsFreq
        fileHeader['End'] /= self.tsFreq
        return fileHeader

    def _ReadFileHeader(self):
        fileHeaderFormat = '<i i 256s d i i i 260s'
        headerBytes = self.theFile.read(struct.calcsize(fileHeaderFormat))
        if len(headerBytes) < struct.calcsize(fileHeaderFormat):
            raise ValueError("unable to read file header")
        fhValues = struct.unpack(fileHeaderFormat, headerBytes)
        keys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'End', 'NumVars', 'Padding']
        fileHeader = dict(zip(keys, fhValues))
        del fileHeader['Padding']

        if fileHeader['MagicNumber'] != 827868494 or fileHeader['Frequency'] <= 0:
            raise ValueError('Invalid .nex file')

        fileHeader['Comment'] = self._ToString(fileHeader['Comment'], True)
        self.tsFreq = fileHeader['Frequency']
        self.fromTicksToSeconds = 1.0 / self.tsFreq
        fileHeader['Beg'] /= self.tsFreq
        fileHeader['End'] /= self.tsFreq
        return fileHeader

    def _ReadVarHeader(self):
        varHeaderFormat = '<i i 64s i i i i i i d d d d i i i d d 52s'
        varHeaderSize = struct.calcsize(varHeaderFormat)
        headerBytes = self.theFile.read(varHeaderSize)
        if len(headerBytes) != varHeaderSize:
            raise ValueError('unable to read variable header') 
        vhValues = struct.unpack(varHeaderFormat, headerBytes)
        keys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'Wire', 'Unit', 'Gain', 'Filter', 'XPos', 'YPos',
                'SamplingRate', 'ADtoMV', 'NPointsWave', 'NMarkers', 'MarkerLength', 'MVOffset', 'PreThrTime', 'Padding']
        varHeader = dict(zip(keys, vhValues))
        del varHeader['Padding']

        varHeader['Name'] = self._ToString(varHeader['Name'], True)
        # add fields that are in nex5 only
        varHeader['TsDataType'] = 0
        varHeader['ContDataType'] = 0
        varHeader['ContFragIndexType'] = 0
        varHeader['MarkerDataType'] = 0
        return varHeader

    def _ReadNex5VarHeader(self):
        varHeaderFormat = '<i i 64s Q Q i i d 32s d d Q d i i i i 60s'
        varHeaderSize = struct.calcsize(varHeaderFormat)
        headerBytes = self.theFile.read(varHeaderSize)
        if len(headerBytes) != varHeaderSize:
            raise ValueError('unable to read variable header') 
        vhValues = struct.unpack(varHeaderFormat, headerBytes)
        keys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'TsDataType', 'ContDataType', 'SamplingRate', 'Units',
                'ADtoMV', 'MVOffset', 'NPointsWave', 'PreThrTime', 'MarkerDataType', 'NMarkers', 'MarkerLength',
                'ContFragIndexType', 'Padding']
        varHeader = dict(zip(keys, vhValues))
        del varHeader['Padding']
        if varHeader['Version'] != 500:
            raise ValueError("invalid variable header version")

        varHeader['Name'] = self._ToString(varHeader['Name'])
        varHeader['Units'] = self._ToString(varHeader['Units'])
        if varHeader['ContDataType'] == 1:
            varHeader['ADtoMV'] = 1
            varHeader['MVOffset'] = 0
        return varHeader

    def _ReadTimestamps(self, var):
        tsValueType = DataFormat.INT32
        if var['Header']['TsDataType'] == 1:
                tsValueType = DataFormat.INT64
        var['Timestamps'] = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)

    def _ReadAndScaleValuesUsingNumpy(self, valueType, count, coeff=1.0, divide=False):
        numpyTypes = [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
        numpyType = numpyTypes[valueType]
        values = np.fromfile(self.theFile, numpyType, count)
        if len(values) != count:
            raise ValueError("unable to read all values")
        if coeff == 1.0:
            return values
        if divide:
            return values / coeff
        else:
            return values * coeff
    
    def _ReadAndScaleValues(self, valueType, count, coeff=1.0, divide=False):
        if self.useNumpy:
            return self._ReadAndScaleValuesUsingNumpy(valueType, count, coeff, divide)
        nb = DataFormat.NumBytesPerItem(valueType)
        vtype = DataFormat.StructTypeFromDataType(valueType)
        valuesBytes = self.theFile.read(nb*count)
        if len(valuesBytes) != nb*count:
            raise ValueError("unable to read all values") 
        x = struct.unpack(vtype*count, valuesBytes)
        valuesAsList = list(x)    
        if coeff == 1.0:
            return valuesAsList
        if divide:
            return [x / coeff for x in valuesAsList]
        else:
            return [x * coeff for x in valuesAsList]

    def _ReadIntervals(self, var):
        if var['Header']['Count'] == 0:
            var['Intervals'] = [[], []]
            return
        tsValueType = DataFormat.INT32
        if var['Header']['TsDataType'] == 1:
            tsValueType = DataFormat.INT64
        intStarts = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        intEnds = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        var['Intervals'] = [intStarts, intEnds]

    def _ReadWaveforms(self, var):
        if var['Header']['Count'] == 0:
            var['WaveformValues'] = []
            return
        if var['Header']['NPointsWave'] <= 0:
            raise ValueError('invalid waveform header: NPointsWave is not positive')
        self._ReadTimestamps(var)
        wfValueType = DataFormat.INT16
        coeff = var['Header']['ADtoMV']
        woffset = var['Header']['MVOffset']
        if var['Header']['ContDataType'] == 1:
            wfValueType = DataFormat.FLOAT32
            coeff = 1.0
            woffset = 0.0
            
        wf = self._ReadAndScaleValues(wfValueType, var['Header']['Count'] * var['Header']['NPointsWave'], coeff)
        if self.useNumpy:
            import numpy as np
            var['WaveformValues'] = wf.reshape( var['Header']['Count'], var['Header']['NPointsWave'])
            if woffset != 0:
                var['WaveformValues'] = var['WaveformValues'] + woffset
        else:
            if woffset != 0:
                var['WaveformValues'] = list(self._Chunks([x + woffset for x in wf], var['Header']['NPointsWave']))
            else:
                var['WaveformValues'] = list(self._Chunks(wf, var['Header']['NPointsWave']))

    def _Chunks(self, theList, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(theList), n):
            yield theList[i:i + n]

    def _ReadPopVectors(self, var):
        var['Weights'] = self._ReadAndScaleValues(DataFormat.FLOAT64, var['Header']['Count'])

    def _ReadContinuous(self, var):
        tsValueType = DataFormat.INT32
        if var['Header']['TsDataType'] == 1:
            tsValueType = DataFormat.INT64
        var['FragmentTimestamps'] = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        indexValueType = DataFormat.UINT32
        if var['Header']['ContFragIndexType'] == 1:
            indexValueType = DataFormat.UINT64
        var['FragmentIndexes'] = self._ReadAndScaleValues(indexValueType, var['Header']['Count'])
        var['FragmentCounts'] = []
        for frag in range(len(var['FragmentIndexes'])):
            if frag < var['Header']['Count'] - 1:
                count = var['FragmentIndexes'][frag+1] - var['FragmentIndexes'][frag]
            else:
                count = var['Header']['NPointsWave'] - var['FragmentIndexes'][frag]
            var['FragmentCounts'].append(count)
            
        coeff = var['Header']['ADtoMV']
        woffset = var['Header']['MVOffset']
        
        contValueType = DataFormat.INT16
        
        if var['Header']['ContDataType'] == 1:
            contValueType = DataFormat.FLOAT32
            coeff = 1.0
            woffset = 0.0
        var['ContinuousValues'] = self._ReadAndScaleValues(contValueType, var['Header']['NPointsWave'], coeff)
        if woffset != 0:
            var['ContinuousValues'] = [x + woffset for x in var['ContinuousValues']]


    def _ReadMarker(self, var):
        self._ReadTimestamps(var)
        var['Fields'] = []
        var['MarkerFieldNames'] = []
        var['Markers'] = []
        isNumeric = True
        for field in range(var['Header']['NMarkers']):
            # marker names and values should be ASCII strings
            field = {'Name': self._ToString(self.theFile.read(64), True).strip()}
            var['MarkerFieldNames'].append(field['Name'])
            if var['Header']['MarkerDataType'] == 0:
                length = var['Header']['MarkerLength']
                field['Markers'] = [self._ToString(self.theFile.read(length), True) for m in range(var['Header']['Count'])]
                if isNumeric:
                    for m in field['Markers']:
                        try:
                            n = int(m)
                        except ValueError:
                            isNumeric = False
                            break

                var['Fields'].append(field)
            else:
                field['Markers'] = self._ReadAndScaleValues(DataFormat.UINT32, var['Header']['Count'])
                var['Fields'].append(field)
        # convert to numbers if all the fields that contain only numbers to have the same values as in nex python interface
        if isNumeric:
            for f in var['Fields']:
                # we can use int (not uint that does not exist in Python anyway) since integers have no limit in Python
                f['Markers'] = [int(m) for m in f['Markers']]
        for f in var['Fields']:
            var['Markers'].append(f['Markers'])


class NexWriter(object):
    """
    Nex file writer class.
    Sample code:

    import nex
    w = nex.NexWriter(100000)
    w.fileData['FileHeader']['Comment'] = 'this is a comment'
    w.AddNeuron('neuron1', [1, 2, 3, 4])
    w.AddContVarWithSingleFragment('cont1', 2, 10000, [5, 6, 7, 8])
    w.WriteNexFile('C:\\Data\\testFileWrittenInPython.nex')
    w.WriteNex5File('C:\\Data\\testFileWrittenInPython.nex5', 1)

    """
    def __init__(self, timestampFrequency, useNumpy=False):
        """
        Constructor
        :param timestampFrequency: timestamp frequency in Hertz. Timestamps are stored as integers representing
                number of ticks, where tick = 1.0/timestampFrequency
        :param useNumpy: option to use numpy
        """
        self.theFile = None
        self.tsFreq = timestampFrequency
        self.useNumpy = useNumpy
        self.fromTicksToSeconds = 1.0/timestampFrequency

        self.varHeaderKeys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'Wire', 'Unit', 'Gain', 'Filter', 'XPos', 'YPos',
                'SamplingRate', 'ADtoMV', 'NPointsWave', 'NMarkers', 'MarkerLength', 'MVOffset', 'PreThrTime', 'Padding']

        self.nex5VarHeaderKeys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'TsDataType', 'ContDataType',
                                  'SamplingRate', 'Units',
                'ADtoMV', 'MVOffset', 'NPointsWave', 'PreThrTime', 'MarkerDataType', 'NMarkers', 'MarkerLength',
                'ContFragIndexType', 'Padding']

        self.fileHeaderKeys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'End', 'NumVars', 'Padding']
        self.fileHeaderKeysToWrite = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'BegTicks', 'EndTicks', 'NumVars', 'Padding']
        self.nex5fileHeaderKeys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'NumVars', 'MetaOffset', 'End',
                'Padding']
        self.nex5fileHeaderKeysToWrite = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'BegTicks', 'NumVars',
                                   'MetaOffset', 'EndTicks', 'Padding']
        # we will add nex5 file header keys (add MetaOffset)
        fhValues = [827868494, 106, '', timestampFrequency, 0, 0, 0, 0, '']
        fileHeader = dict(zip(self.nex5fileHeaderKeys, fhValues))
        self.fileData = {'FileHeader': fileHeader, 'Variables': []}

    def AddNeuron(self, name, timestamps, wire=0, unit=0, xpos=0, ypos=0):
        """
        Adds neuron file variable
        :param name: neuron name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param wire: wire (electrode) number
        :param unit: unit number
        :param xpos: x position in [0, 100] range (used in 3d displays)
        :param ypos: y position in [0, 100] range (used in 3d displays)
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        vhValues = [NexFileVarType.NEURON, 100, name, 0, len(timestamps), wire, unit, 0, 0, xpos, ypos, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        self.fileData['Variables'].append(var)

    def AddEvent(self, name, timestamps):
        """
        Adds event file variable
        :param name: event name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        vhValues = [NexFileVarType.EVENT, 100, name, 0, len(timestamps), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        self.fileData['Variables'].append(var)

    def AddIntervalVariable(self, name, intStarts, intEnds):
        """
        Adds interval variable to file data
        :param name: variable name
        :param intStarts: list interval starts in seconds or numpy array if numpy option is specified in constructor
        :param intEnds: list interval ends in seconds or numpy array if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Interval starts', intStarts)
            self._VerifyIsNumpyArray('Interval ends', intEnds)
        vhValues = [NexFileVarType.INTERVAL, 100, name, 0, len(intStarts), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Intervals'] = [intStarts, intEnds]
        self.fileData['Variables'].append(var)

    def AddContVarWithSingleFragment(self, name, timestampOfFirstDataPoint, SamplingRate, values):
        """
        Adds continuous variable with a single fragment
        :param name: variable name
        :param timestampOfFirstDataPoint: time of first data point in seconds
        :param SamplingRate: sampling rate in Hz
        :param values: list of variable values in mV or numpy array of values if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Values', values)
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in continuous')
        vhValues = [NexFileVarType.CONTINUOUS, 100, name, 0, 1, 0, 0, 0, 0, 0, 0, SamplingRate, 1.0,
                    len(values), 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = [timestampOfFirstDataPoint]
        var['FragmentIndexes'] = [0]
        var['FragmentCounts'] = [len(values)]
        if self.useNumpy:
            import numpy as np
            var['Timestamps'] = np.array(var['Timestamps'])
            var['FragmentIndexes'] = np.array(var['FragmentIndexes'])
            var['FragmentCounts'] = np.array(var['FragmentCounts'])
        var['ContinuousValues'] = values
        self.fileData['Variables'].append(var)

    def AddContVarWithMultipleFragments(self, name, timestamps, SamplingRate, fragmentValues):
        """
        Adds continuous variable with multiple fragments
        :param name: variable name
        :param timestamps: list fragment start times in seconds or numpy array of fragment start times
                     if numpy option is specified in constructor
        :param SamplingRate: sampling rate in Hz
        :param fragmentValues: list of lists, each sublist is array data values in mV (or list of numpy arrays)
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
            for fragment in fragmentValues:
                self._VerifyIsNumpyArray('FragmentValues', fragment)
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in continuous')
        if len(timestamps) != len(fragmentValues):
            raise ValueError('should be the same number of timestamps and fragments')
        totalValues = 0
        var = {'FragmentIndexes': [], 'FragmentCounts': [], 'ContinuousValues': []}
        for fragment in fragmentValues:
            var['FragmentIndexes'].append(totalValues)
            var['FragmentCounts'].append(len(fragment))
            if not self.useNumpy:
                var['ContinuousValues'].extend(fragment)
            totalValues += len(fragment)
        if self.useNumpy:
            import numpy as np
            var['ContinuousValues'] = np.array([])
            for fragment in fragmentValues:
                var['ContinuousValues'] = np.append(var['ContinuousValues'], fragment)

        vhValues = [NexFileVarType.CONTINUOUS, 100, name, 0, 1, 0, 0, 0, 0, 0, 0, SamplingRate, 1.0,
                    totalValues, 0, 0, 0, 0, '']
        var['Header'] = dict(zip(self.varHeaderKeys, vhValues))
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps

        if self.useNumpy:
            import numpy as np
            var['FragmentIndexes'] = np.array(var['FragmentIndexes'])
            var['FragmentCounts'] = np.array(var['FragmentCounts'])

        self.fileData['Variables'].append(var)

    def AddMarker(self, name, timestamps, fieldNames, markerFields):
        """
        Adds marker variable.
        :param name: variable name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param fieldNames: list of field names
        :param markerFields: a list of lists (one list of numbers or strings per marker field)
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        if len(fieldNames) != len(markerFields):
            raise ValueError('different number of field names and field values')
        for x in markerFields:
            if len(x) != len(timestamps):
                raise ValueError('different number of timestamps and field values in a single field')
        vhValues = [NexFileVarType.MARKER, 100, name, 0, len(timestamps), 0, 0, 0, 0, 0, 0, 0, 0, 0, len(fieldNames), 6, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Header']['MarkerDataType'] = 1
        var['Timestamps'] = timestamps
        var['MarkerFieldNames'] = fieldNames
        var['Markers'] = markerFields
        self._CalcMarkerLength(var)
        self.fileData['Variables'].append(var)

    def AddWave(self, name, timestamps, SamplingRate, WaveformValues, NPointsWave=0, PrethresholdTimeInSeconds=0, wire=0, unit=0):
        """
        Adds waveform variable.
        :param name: variable name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param SamplingRate: sampling rate of w/f values in Hz
        :param WaveformValues: a list of lists, each sublist contains values of a single waveform in mV;
               if numpy option is specified in constructor, numpy matrix
        :param NPointsWave: number of data points in each wave
        :param PrethresholdTimeInSeconds: pre-threshold time in seconds
        :param wire: wire (electrode) number
        :param unit: unit number
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
            self._VerifyIsNumpyArray('WaveformValues', WaveformValues)
        if len(timestamps) != len(WaveformValues):
            raise ValueError('different number of timestamps and number of waveforms')
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in wave')
        var = {}
        if len(WaveformValues) > 0 and len(WaveformValues[0]) > 0:
            NPointsWave = len(WaveformValues[0])
        if NPointsWave == 0:
            raise ValueError('invalid number of data points in wave')
        vhValues = [NexFileVarType.WAVEFORM, 100, name, 0, len(timestamps), wire, unit, 0, 0, 0, 0, SamplingRate, 1.0, NPointsWave, 0, 0, 0, PrethresholdTimeInSeconds, '']
        var['Header'] = dict(zip(self.varHeaderKeys, vhValues))
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        var['WaveformValues'] = WaveformValues
        self.fileData['Variables'].append(var)

    def WriteNexFile(self, filePath):
        """
        Writes file data as .nex file.
        :param filePath: full path of file
        :return: none
        """
        self.theFile = open(filePath, 'wb')
        self.fileData['FileHeader']['MagicNumber'] = 827868494
        self.fileData['FileHeader']['NexFileVersion'] = 106
        numberOfVariables = len(self.fileData['Variables'])
        self.fileData['FileHeader']['NumVars'] = numberOfVariables

        maxTs = self._MaximumTimestamp()
        if round(maxTs * self.tsFreq) > pow(2, 31):
            raise ValueError('unable to save as .nex file: max timestamp exceeds 32-bit range; you can save as .nex5 file instead')
        self.fileData['FileHeader']['BegTicks'] = int(round(self.fileData['FileHeader']['Beg'] * self.tsFreq))
        self.fileData['FileHeader']['EndTicks'] = int(round(maxTs * self.tsFreq))

        for v in self.fileData['Variables']:
            v['Header']['TsDataType'] = 0
            v['Header']['Version'] = 102
            v['Header']['ContDataType'] = 0
            v['Header']['MarkerDataType'] = 0
            self._CalculateScaling(v)
            self._CalcMarkerLength(v)

        dataOffset = 544 + numberOfVariables*208
        for v in self.fileData['Variables']:
            v['Header']['Count'] = self._VarCount(v)
            v['Header']['DataOffset'] = dataOffset
            dataOffset += self._VarNumDataBytes(v)

        fileHeaderFormat = '<i <i 256s <d <i <i <i 260s'.split()
        for i in range(len(self.fileHeaderKeysToWrite)):
            self._WriteField(fileHeaderFormat[i], self.fileData['FileHeader'][self.fileHeaderKeysToWrite[i]])

        varHeaderFormat = '<i <i 64s <i <i <i <i <i <i <d <d <d <d <i <i <i <d <d 52s'.split()
        for v in self.fileData['Variables']:
            for i in range(len(self.varHeaderKeys)):
                self._WriteField(varHeaderFormat[i], v['Header'][self.varHeaderKeys[i]])

        for v in self.fileData['Variables']:
            self._VarWriteData(v)

        self.theFile.close()

    def WriteNex5File(self, filePath, saveContValuesAsFloats=0):
        """
        Writes file data as .nex5 file.
        :param filePath: full path of file
        :param saveContValuesAsFloats: if zero, continuous values are saved as 16-bit integers; if 1, saved as floats
        :return:
        """
        self.theFile = open(filePath, 'wb')
        self.fileData['FileHeader']['MagicNumber'] = 894977358
        self.fileData['FileHeader']['NexFileVersion'] = 501
        nvars = len(self.fileData['Variables'])
        self.fileData['FileHeader']['NumVars'] = nvars

        maxTs = self._MaximumTimestamp()
        tsAs64 = 0
        if round(maxTs * self.tsFreq) > pow(2, 31):
            tsAs64 = 1
            self.fileData['FileHeader']['NexFileVersion'] = 502

        for v in self.fileData['Variables']:
            v['Header']['TsDataType'] = tsAs64
            v['Header']['Version'] = 500
            v['Header']['ContDataType'] = saveContValuesAsFloats
            if v['Header']['Type'] == NexFileVarType.MARKER:
                self._CalcMarkerLength(v)
                if v['AllNumbers']:
                    v['Header']['MarkerDataType'] = 1
                else:
                    v['Header']['MarkerDataType'] = 0
            self._CalculateScaling(v)

        self.fileData['FileHeader']['BegTicks'] = int(round(self.fileData['FileHeader']['Beg'] * self.tsFreq))
        self.fileData['FileHeader']['EndTicks'] = int(round(maxTs * self.tsFreq))

        dataOffset = 356 + nvars * 244
        for v in self.fileData['Variables']:
            v['Header']['Count'] = self._VarCount(v)
            v['Header']['DataOffset'] = dataOffset
            dataOffset += self._VarNumDataBytes(v)

        fileHeaderFormat = '<i <i 256s <d <q <i <Q <q 56s'.split()
        for i in range(len(self.nex5fileHeaderKeysToWrite)):
            self._WriteField(fileHeaderFormat[i], self.fileData['FileHeader'][self.nex5fileHeaderKeysToWrite[i]])

        varHeaderFormat = '<i <i 64s <Q <Q <i <i <d 32s <d <d <Q <d <i <i <i <i 60s'.split()
        for v in self.fileData['Variables']:
            for i in range(len(self.nex5VarHeaderKeys)):
                self._WriteField(varHeaderFormat[i], v['Header'][self.nex5VarHeaderKeys[i]])

        for v in self.fileData['Variables']:
            self._VarWriteData(v)

        metaData = {"file": {}, 'variables': []}
        metaData["file"]["writerSoftware"] = {}
        metaData["file"]["writerSoftware"]["name"] = 'nexfile.py'
        metaData["file"]["writerSoftware"]["version"] = 'Sep-11-2019'
        for v in self.fileData['Variables']:
            varMeta = {'name': v['Header']['Name']}
            if v['Header']['Type'] == NexFileVarType.NEURON or v['Header']['Type'] == NexFileVarType.WAVEFORM:
                varMeta['unitNumber'] = v['Header']['Unit']
                varMeta['probe'] = {}
                varMeta['probe']['wireNumber'] = v['Header']['Wire']
                varMeta['probe']['position'] = {}
                varMeta['probe']['position']['x'] = v['Header']['XPos']
                varMeta['probe']['position']['y'] = v['Header']['YPos']
            metaData['variables'].append(varMeta)

        metaString = json.dumps(metaData).encode('utf-8')
        pos = self.theFile.tell()
        self.theFile.write(metaString)
        metaPosInHeader = 284
        self.theFile.seek(metaPosInHeader, 0)
        self.theFile.write(struct.pack('<Q', pos))

        self.theFile.close()

    # the following class methods are internal
    def _VerifyIsNumpyArray(self, name, a):
        import numpy as np
        if not isinstance(a, np.ndarray):
            raise ValueError(name + ' should be a numpy array')

    def _ConvertStringToBytesIfNeeded(self, stringOrBytes):
        if isinstance(stringOrBytes, bytes):
            return stringOrBytes
        else:
            return stringOrBytes.encode('utf-8')

    def _WriteField(self, theFormat, theField):
        if theFormat.endswith('s'):
            theField = self._ConvertStringToBytesIfNeeded(theField)
        self.theFile.write(struct.pack(theFormat, theField))
        
    def _WriteList(self, theList, valueType):
        vtype = DataFormat.StructTypeFromDataType(valueType)
        self.theFile.write(struct.pack(vtype*len(theList), *theList))


    def _AddNex5VarHeaderFields(self, var):
        """
        Adds .nex5 variable header fields.
        :param var: file data variable
        :return: none
        """
        var['Header']['TsDataType'] = 0
        var['Header']['ContDataType'] = 0
        var['Header']['Units'] = ''
        var['Header']['MarkerDataType'] = 0
        var['Header']['ContFragIndexType'] = 0

    def _BytesInTimestamp(self, var):
        """
        Calculates number of bytes in timestamp.
        :param var: file data variable
        :return: number of bytes in timestamp
        """
        if var['Header']['TsDataType'] == 0:
            return 4
        else:
            return 8

    def _BytesInContValue(self, var):
        """
        :param var:  file data variable
        :return: number of bytes in continuous value
        """
        if var['Header']['ContDataType'] == 0:
            return 2
        else:
            return 4

    def _VarNumDataBytes(self, var):
        """
        :param var: file data variable
        :return: number of bytes in variable data
        """
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            return self._BytesInTimestamp(var) * len(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return 2 * self._BytesInTimestamp(var) * len(var['Intervals'][0])
        elif varType == NexFileVarType.WAVEFORM:
            return self._BytesInTimestamp(var) * len(var['Timestamps']) + self._BytesInContValue(var) * len(var['Timestamps']) * var['Header']['NPointsWave']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            # population vectors are not supported in this writer class
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            return (self._BytesInTimestamp(var) + 4) * len(var['Timestamps']) + self._BytesInContValue(var) * len(var['ContinuousValues'])
        elif varType == NexFileVarType.MARKER:
            if var['Header']['MarkerDataType'] == 0:
                singleMarkerLength = var['Header']['MarkerLength']
            else:
                singleMarkerLength = 4
            nts = len(var['Timestamps'])
            nm = len(var['MarkerFieldNames'])
            return self._BytesInTimestamp(var) * nts + nm * (64 + nts * singleMarkerLength)

    def _VarCount(self, var):
        """
        Calculates count field for file variables
        :param var:
        :return:
        """
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return len(var['Intervals'][0])
        elif varType == NexFileVarType.WAVEFORM:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.MARKER:
            return len(var['Timestamps'])

    def _MaxOfNumpyArrayOrZero(self, x):
        import numpy as np
        if len(x) == 0:
            return 0
        else:
            return np.max(x)

    def _VarMaxTimestampNumpy(self, var):
        import numpy
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT or varType == NexFileVarType.MARKER:
            return self._MaxOfNumpyArrayOrZero(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return self._MaxOfNumpyArrayOrZero(var['Intervals'][1])
        elif varType == NexFileVarType.WAVEFORM:
            if len(var['Timestamps']) == 0:
                return 0
            return self._MaxOfNumpyArrayOrZero(var['Timestamps']) + (var['Header']['NPointsWave']-1)/var['Header']['SamplingRate']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            if len(var['Timestamps']) == 0:
                return 0
            return var['Timestamps'][-1] + (var['FragmentCounts'][-1] - 1) / var['Header']['SamplingRate']

    def _MaxValueOrZero(self, theList):
        if len(theList) > 0:
            return max(theList)
        return 0

    def _VarMaxTimestamp(self, var):
        if self.useNumpy:
            return self._VarMaxTimestampNumpy(var)
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT or varType == NexFileVarType.MARKER:
            return self._MaxValueOrZero(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return self._MaxValueOrZero(var['Intervals'][1])
        elif varType == NexFileVarType.WAVEFORM:
            if len(var['Timestamps']) == 0:
                return 0
            return max(var['Timestamps']) + (var['Header']['NPointsWave']-1)/var['Header']['SamplingRate']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            if len(var['Timestamps']) == 0:
                return 0
            return var['Timestamps'][-1] + (var['FragmentCounts'][-1] - 1) / var['Header']['SamplingRate']

    def _VarWriteTimestampsNumpy(self, var, timestamps):
        import numpy as np
        if self._BytesInTimestamp(var) == 4:
            np.round(timestamps * self.tsFreq).astype(np.int32).tofile(self.theFile)
        else:
            np.round(timestamps * self.tsFreq).astype(np.int64).tofile(self.theFile)

    def _VarWriteTimestamps(self, var, timestamps):
        if self.useNumpy:
            return self._VarWriteTimestampsNumpy(var, timestamps)
        if self._BytesInTimestamp(var) == 4:
            tsTicks = [int(round(x * self.tsFreq)) for x in timestamps]
            self._WriteList(tsTicks, DataFormat.INT32)
        else:
            tsTicks = [int(round(x * self.tsFreq)) for x in timestamps]
            self._WriteList(tsTicks, DataFormat.INT64)

    def _VarWriteWaveformsNumpy(self, var):
        import numpy as np
        if self._BytesInContValue(var) == 2:
            np.round(var['WaveformValues'] / var['Header']['ADtoMV']).astype(np.int16).tofile(self.theFile)
        else:
            var['WaveformValues'].astype(np.float32).tofile(self.theFile)

    def _VarWriteContinuousValuesNumpy(self, var):
        import numpy as np
        if self._BytesInContValue(var) == 2:
            np.round(var['ContinuousValues'] / var['Header']['ADtoMV']).astype(np.int16).tofile(self.theFile)
        else:
            var['ContinuousValues'].astype(np.float32).tofile(self.theFile)
            
    def _WriteWaveformVarData(self,var):
        self._VarWriteTimestamps(var, var['Timestamps'])
        if self.useNumpy:
            self._VarWriteWaveformsNumpy(var)
            return
        if self._BytesInContValue(var) == 2:
            for w in var['WaveformValues']:
                waveValues = [int(x / var['Header']['ADtoMV']) for x in w]
                self._WriteList(waveValues, DataFormat.INT16)
        else:
            for w in var['WaveformValues']:
                self._WriteList(w, DataFormat.FLOAT32)
                
    def _WriteContinuousVarData(self,var):
        self._VarWriteTimestamps(var, var['Timestamps'])
        if self.useNumpy:
            import numpy as np
            var['FragmentIndexes'].astype(np.uint32).tofile(self.theFile)
            self._VarWriteContinuousValuesNumpy(var)
        else:
            self._WriteList(var['FragmentIndexes'], DataFormat.UINT32)
            if self._BytesInContValue(var) == 2:
                contValues = [int(x / var['Header']['ADtoMV']) for x in var['ContinuousValues']]
                self._WriteList(contValues, DataFormat.INT16)
            else:
                self._WriteList(var['ContinuousValues'], DataFormat.FLOAT32)

    def _VarWriteData(self, var):
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            self._VarWriteTimestamps(var, var['Timestamps'])
            return
        elif varType == NexFileVarType.INTERVAL:
            for i in range(2):
                self._VarWriteTimestamps(var, var['Intervals'][i])
            return
        elif varType == NexFileVarType.WAVEFORM:
            self._WriteWaveformVarData(var)
            return
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return
        elif varType == NexFileVarType.CONTINUOUS:
            self._WriteContinuousVarData(var)
            return
        elif varType == NexFileVarType.MARKER:
            self._VarWriteTimestamps(var, var['Timestamps'])
            for i, name in enumerate(var['MarkerFieldNames']):
                self._WriteField('64s', name)
                if var['Header']['MarkerDataType'] == 0:
                    for v in var['Markers'][i]:
                        # convert number to string if needed first
                        if isinstance(v, numbers.Number):
                            sv = '{0:05d}'.format(v) + '\x00'
                        else:
                            sv = v
                            while len(sv) < var['Header']['MarkerLength']:
                                sv += '\x00'
                        self.theFile.write(sv.encode('utf-8'))
                else:
                    self._WriteList(var['Markers'][i], DataFormat.UINT32)
            return

    def _CalcMarkerLength(self, var):
        if var['Header']['Type'] != NexFileVarType.MARKER:
            return
        maxStringLength = 0
        allNumbers = True
        for field in var['Markers']:
            for x in field:
                if not isinstance(x, numbers.Number):
                    allNumbers = False
                    if not isinstance(x, str):
                        raise ValueError('marker values should be either numbers or strings')
                    maxStringLength = max(maxStringLength, len(x))
        var['AllNumbers'] = allNumbers
        if allNumbers:
            var['Header']['MarkerLength'] = 6
        else:
            var['Header']['MarkerLength'] = max(6, maxStringLength + 1)

    def _MaximumTimestamp(self):
        maxTs = 0
        for v in self.fileData['Variables']:
            maxTs = max(maxTs, self._VarMaxTimestamp(v))
        return maxTs

    def _SignalAbsMaxNumPy(self, signal):
        import numpy as np
        return np.max(np.abs(signal))

    def _CalculateScaling(self, var):
        varType = var['Header']['Type']
        if varType == NexFileVarType.CONTINUOUS:
            if var['Header']['ContDataType'] == 1:
                var['Header']['ADtoMV'] = 1
                return
            contMax = 0
            if self.useNumpy:
                contMax = self._SignalAbsMaxNumPy(var['ContinuousValues'])
            else:
                contMax = max(contMax, abs(max(var['ContinuousValues'])))
                contMax = max(contMax, abs(min(var['ContinuousValues'])))
            if contMax == 0:
                var['Header']['ADtoMV'] = 1
            else:
                var['Header']['ADtoMV'] = contMax / 32767.0
            return
        if varType == NexFileVarType.WAVEFORM:
            if var['Header']['ContDataType'] == 1:
                var['Header']['ADtoMV'] = 1
                return
            waveMax = 0
            if self.useNumpy:
                waveMax = self._SignalAbsMaxNumPy(var['WaveformValues'])
            else:
                for w in var['WaveformValues']:
                    waveMax = max(waveMax, abs(max(w)))
                    waveMax = max(waveMax, abs(min(w)))
            if waveMax == 0:
                var['Header']['ADtoMV'] = 1
            else:
                var['Header']['ADtoMV'] = waveMax / 32767.0
