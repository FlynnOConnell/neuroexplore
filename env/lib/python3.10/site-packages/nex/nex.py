from __future__ import annotations
import json
import os
import numbers
import csv
import sys
import random
import math
from typing import List
import numpy as np
import builtins
from io import StringIO
from . import nexLink



def GetListOfNumbers(varWithListOfNumbers, varName):
    """ Make sure that passed parameter can be converted to a list of numbers. """
    try:
        theList = list(varWithListOfNumbers)
        for item in theList:
            f = float(item)
        return theList
    except:
        error = f'parameter "{varName}" should be either a list of numbers, array or numpy array'
        raise ValueError(error)


class NexVar:
    """ Class representing a data channel (a variable) in a data file. """
    def __init__(self, spec):
        self.spec = spec
        self.varId = int(spec['value'])
        
    def __eq__(self, otherId):
        return self.varId == otherId
    
    def __ne__(self, otherId):
        return self.varId != otherId

    def __bool__(self):
        return self.varId != 0

    __nonzero__ = __bool__

    def __repr__(self):
        if self.varId > 0:
            return 'NexVar: ' + self.Name() + ": " + str(self.spec)
        else:
            return 'NexVar: ' + str(self.spec)
        
    def __getitem__(self, index):
        return nexLink.nexGetIndexedValue(self.varId, index)

    def __setitem__(self, index, theValue):
        nexLink.nexSetIndexedValue(self.varId, index, theValue)

    def __add__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContAdd(self, other)
        elif isinstance(other, NexVar):
            return ContAddCont(self, other)
        raise ValueError('invalid operand in NexVar addition. Number or NexVar is expected')
    
    def __radd__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContAdd(self, other)
        elif isinstance(other, NexVar):
            return ContAddCont(self, other)
        raise ValueError('invalid operand in NexVar addition. Number or NexVar is expected')

    def __sub__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContAdd(self, -other)
        elif isinstance(other, NexVar):
            return ContSubtractCont(self, other)
        raise ValueError('invalid operand in NexVar subtraction. Number or NexVar is expected')

    def __rsub__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContAdd(self, -other)
        elif isinstance(other, NexVar):
            return ContSubtractCont(self, other)
        raise ValueError('invalid operand in NexVar subtraction. Number or NexVar is expected')
    
    def __mul__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContMult(self, other)
        elif isinstance(other, NexVar):
            return ContMultCont(self, other)
        raise ValueError('invalid operand in NexVar multiplication. Number or NexVar is expected')

    def __rmul__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            return ContMult(self, other)
        elif isinstance(other, NexVar):
            return ContMultCont(self, other)
        raise ValueError('invalid operand in NexVar multiplication. Number or NexVar is expected')

    def __div__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            if float(other) == 0.0:
                raise ValueError('invalid operand in NexVar division. Non-zero number is expected')
            return ContMult(self, 1.0/float(other))
        elif isinstance(other, NexVar):
            return ContDivCont(self, other)
        raise ValueError('invalid operand in NexVar division. Number or NexVar is expected')
    
    def __truediv__(self, other) -> NexVar:
        if isinstance(other, numbers.Number):
            if float(other) == 0.0:
                raise ValueError('invalid operand in NexVar division. Non-zero number is expected')
            return ContMult(self, 1.0/float(other))
        elif isinstance(other, NexVar):
            return ContDivCont(self, other)
        raise ValueError('invalid operand in NexVar division. Number or NexVar is expected')

    def _Get(self, propertyName):
        """Internal method. Returns variable property."""
        return nexLink.nexGetProperty(1, self.varId, propertyName)
    
    def Name(self) -> str:
        """Returns variable name."""
        return self._Get('Name')   
    
    def SamplingRate(self) -> float:
        """Returns continuous variable sampling rate."""
        return self._Get('SamplingRate')   

    def NumPointsInWave(self) -> int:
        """Returns the number of data points in a wave for a waveform variable."""
        return int(self._Get('NumPointsInWave'))   

    def Metadata(self) -> dict:
        """Returns variable metadata."""
        # error processing is already done in Get
        resultJsonString = self._Get('Metadata')
        return json.loads(resultJsonString)

    def Timestamps(self) -> List[float]:
        """Returns variable timestamps (in seconds) as a list."""
        return self._Get('Timestamps')   

    def ContinuousValues(self) -> List[float]:
        """Returns all values of a continuous variable in milliVolts."""
        return self._Get('ContinuousValues')

    def ContinuousValuesAsInt16(self) -> List[float]:
        """Returns raw values of a continuous variable as a list of int16 integers converted to floats."""
        return self._Get('ContinuousValuesAsInt16')
    
    def ContMin(self) -> float:
        """Returns minimum of all values of a continuous variable."""
        return self._Get('ContMin')
    
    def ContMax(self) -> float:
        """Returns maximum of all values of a continuous variable."""
        return self._Get('ContMax')

    def ContMean(self) -> float:
        """Returns average of all values of a continuous variable."""
        return self._Get('ContMean')
    
    def FragmentTimestamps(self) -> List[float]:
        """Returns fragment timestamps (in seconds) of a continuous variable."""
        return self._Get('FragmentTimestamps')

    def FragmentCounts(self) -> List[float]:
        """Returns fragment counts of a continuous variable."""
        return self._Get('FragmentCounts')

    def Intervals(self) -> List[List[float]]:
        """
        Returns intervals of an interval variable as a list.
        The first element of the returned list is the list of all interval start times.
        The second element of the returned list is the list of all interval end times.
        """
        return self._Get('Intervals') 
      
    def WaveformValues(self) -> List[List[float]]:
        """
        Returns waveforms of a waveform variable as a list of lists.
        The first element of the returned list contains the list of values of the first waveform, 
        the second - values of the second waveform, etc.
        """
        return self._Get('WaveformValues')
       
    def Markers(self) -> List[List]:
        """
        Returns marker values of a marker variable as a list of lists.
        The first element of the returned list contains values of the first marker field, 
        the second - values of the second marker field, etc.
        If all marker values cannot be converted to numbers, the returned values are strings.
        Otherwise, the returned values are numbers.
        """
        return self._Get('Markers')   

    def MarkerFieldNames(self) -> List[str]:
        """Returns field names of a marker variable."""
        return self._Get('MarkerFieldNames')
    
    def SetContVarTimestampsAndValues(self, timestamps:List[float], values:List[float]):
        """Assigns timestamps (in seconds) and values (in milliVolts) of a continuous variable."""
        timestamps = GetListOfNumbers(timestamps, 'timestamps')
        values = GetListOfNumbers(values, 'values')
        nexLink.nexSetContValues( self.varId, timestamps, values)
        
    def SetContVarStartTimeAndValues(self, startTime:float, values:List[float]):
        """Sets start time (in seconds) and continuous values (in milliVolts) of a continuous variable with a single fragment."""
        callPars = {}
        callPars['functionName'] = 'SetContVarStartTimeAndValues'
        callPars['parameters'] = [{'varId':self.varId}, {'start':startTime}]
        # converting int16 to doubles straight from np.ndarray does not work. list, however, works fine for any number type.
        values = GetListOfNumbers(values, 'values')
        _JsonPlusNumList(json.dumps(callPars), values)

    def SetContVarStartTimeAndValues16bit(self, startTime:float, values:List[float]):
        """Sets start time (in seconds) and 16-bit continuous values (as floats) of a continuous variable with a single fragment."""
        callPars = {}
        callPars['functionName'] = 'SetContVarStartTimeAndValues16bit'
        callPars['parameters'] = [{'varId':self.varId}, {'start':startTime}]
        # converting int16 to doubles straight from np.ndarray does not work. list, however, works fine for any number type.
        values = GetListOfNumbers(values, 'values')
        intValues = [int(i) for i in values]
        _JsonPlusNumList(json.dumps(callPars), intValues)

    def SetTimestamps(self, timestamps:List[float]):
        """Sets timestamps (in seconds) of a neuron or event variable."""
        timestamps = GetListOfNumbers(timestamps, 'timestamps')
        nexLink.nexSetTimestamps( self.varId, timestamps) 


class NexDoc:
    """ Class representing document (data file). """
    def __init__(self, spec):
        self.spec = spec
        self.docId = int(spec['value'])
        self.vars = dict()
        
    def __repr__(self):
        return 'NexDoc: ' + str(self.spec)
         
    def __getitem__(self, index:str) -> NexVar:
        if index in self.vars:
            return self.vars[index]
        else:
            theVar = GetVarByName(self.spec, index)
            # store variable only if it is a valid doc var
            if theVar:
                self.vars[index] = theVar
            return theVar
            
    def __setitem__(self, index:str, theValue:NexVar):
        if isinstance(theValue, NexVar):
            nexLink.nexCopyValuesToDocVar(self.docId, index, theValue.varId)
            try:
                newVar = GetVarByName(self.spec, index)
            except Exception as ex:
                s = str(ex)
                if 'document is not valid or cannot be used in demo mode' in s:
                    raise ValueError('unable to perform variable assignment: document is not valid or cannot be used in demo mode')
                else:
                    raise ValueError('invalid variable name "' + index + '"')
        else:
            raise ValueError('invalid right-hand side in assignment')
            
    def __gt__(self, other):
        return self.docId > other
    
    def __eq__(self, otherId):
        return self.docId == otherId
    
    def __ne__(self, otherId):
        return self.docId != otherId

    def __bool__(self):
        return self.docId != 0

    __nonzero__ = __bool__
    
    def _CheckIfValidDoc(self):
        """Internal method. Checks if document is valid and throws error if document is not valid."""
        if not self:
            raise ValueError('Invalid document. Is data file opened in NeuroExplorer?')


    def _OnDeleteVar(self, var:NexVar):
        """Internal method. Called when var is deleted."""
        varName = var.Name()
        if varName in self.vars.keys():
            del self.vars[varName]
            
    def _OnRenameVar(self, var:NexVar, newName:str):
        """Internal method. Called when var is renamed."""
        varName = var.Name()
        if varName in self.vars.keys():
            del self.vars[varName]
        self.vars[newName] = var
        
    def _Get(self, propertyName:str):
        """Internal method. Returns document property."""
        return nexLink.nexGetProperty(0, self.docId, propertyName)

    #anres
    def GetNumResNCols(self) -> int:
        """Returns the number of columns in numerical results."""
        self._CheckIfValidDoc()
        return int(self._Get('NumResNCols'))

    #anres
    def GetNumResNRows(self) -> int:
        """Returns the number of rows in numerical results."""
        self._CheckIfValidDoc()
        return int(self._Get('NumResNRows'))

    #anres
    def GetNumResValue(self, row:int, col:int) -> float:
        """Returns specified numerical results value."""
        # the recommendation is to use doc.GetAllNumericalResults(). see GetNumRes function.
        self._CheckIfValidDoc()
        return nexLink.nexGetIndexedNumResValue(self.docId, row, col)

    #anres
    def GetAllNumericalResults(self) -> List[List[float]]:
        """
        Returns all numerical results as a list of lists.
        The first element contains the values of the first column of numerical results.
        The second element contains values of the second column of numerical results, etc.
        """
        self._CheckIfValidDoc()
        return self._Get('NumRes')

    #anres
    def GetAllNumericalResultsAsCOM(self) -> List[List[float]]:
        """
        Returns all numerical results in the same format as COM method. 
        Matrix of results is transposed compared to GetAllNumericalResults() method.
        """
        self._CheckIfValidDoc()
        res = self._Get('NumRes')
        return map(list, zip(*res))

    #anres
    def GetNumResColumnNames(self) -> List[str]:
        """Returns the list of all column names in numerical results."""
        self._CheckIfValidDoc()
        return self._Get('NumResColumnNames')

    #anres
    def GetNumResSummaryNCols(self) -> int:
        """Returns the number of columns in numerical results summary."""
        return int(self._Get('NumResSummaryNCols'))

    #anres
    def GetNumResSummaryNRows(self) -> int:
        """ Returns the number of rows in numerical results summary."""
        self._CheckIfValidDoc()
        return int(self._Get('NumResSummaryNRows'))

    #anres
    def GetAllNumResSummaryData(self) -> List[List[str]]:
        """
        Returns all values of numerical results summary as a list of lists.
        The first element contains the values of the first column of numerical results summary.
        The second contains values of the second column of numerical results summary, etc.
        The values are returned as strings.
        """
        self._CheckIfValidDoc()
        return self._Get('NumResSummary')

    def GetNumResSummaryColumnNames(self) -> List[str]:
        """Returns the list of all column names in numerical results summary."""
        self._CheckIfValidDoc()
        return self._Get('NumResSummaryColumnNames')
    
    #analysis
    def GetPostProcessingScriptParameter(self) -> str:
        """Returns post-processing script parameter as string."""
        self._CheckIfValidDoc()
        return self._Get('PostProcessingScriptParameter')

    #analysis
    def GetPythonAnalysisInput(self) -> str:
        """Returns Python analysis input as a JSON string."""
        self._CheckIfValidDoc()
        return self._Get('PythonAnalysisInput')

    #analysis
    def SetPythonAnalysisOutput(self, theOutput:str):
        """Sets Python analysis output as a JSON string."""
        self._CheckIfValidDoc()
        return self._SetProperty('PythonAnalysisOutput', theOutput)

    #analysis
    def GetAllAnalysisParameters(self) -> str:
        """Returns all analysis parameters as JSON string consisting of pairs {parName : parValueAsString}."""
        return self._Get('AllAnalysisParameters')
    
    #varselection
    def GetSelVarNames(self) -> List[str]:
        """Returns the list of selected variable names."""
        self._CheckIfValidDoc()
        return self._Get('SelVarNames')

    #docproperties
    def _SetProperty(self, propertyName:str, propertyValue):
        """Internal method. Sets NexDoc property."""
        prop = {'name': propertyName, 'value': propertyValue}
        return nexLink.nexSetDocProperty(self.docId, json.dumps(prop))

    #creatingnewvars
    def CreateWaveformVariable(self, name:str, waveformSamplingRate:float, timestamps:List[float], values:List[List[float]]):
        """Creates new waveform variable with the specified values."""
        self._CheckIfValidDoc()
        timestamps = GetListOfNumbers(timestamps, 'timestamps')
        nexLink.nexCreateWaveformVar(self.docId, name, waveformSamplingRate, timestamps, values) 

    #modvardata
    def SetNeuronPosition(self, neuron:NexVar, x:float, y:float):
        """Sets position of a neuron variable."""
        self._CheckIfValidDoc()
        theValue = {}
        theValue['varId'] = neuron.varId
        theValue['position'] = [x,y]
        self._SetProperty('NeuronPosition', theValue)

    #modvardata
    def SetNeuronWire(self, neuron:NexVar, wire:int):
        """Sets wire (electrode number) of a neuron variable."""
        self._CheckIfValidDoc()
        theValue = {}
        theValue['varId'] = neuron.varId
        theValue['wire'] = wire
        self._SetProperty('NeuronWire', theValue)

    #modvardata
    def SetNeuronUnit(self, neuron:NexVar, unit:int):
        """Sets unit (cluster id) of a neuron variable."""
        self._CheckIfValidDoc()
        theValue = {}
        theValue['varId'] = neuron.varId
        theValue['unit'] = unit
        self._SetProperty('NeuronUnit', theValue)

    def _VarNames(self, varType:str) -> List[str]:
        """Internal helper function. Returns the list of variable names."""
        names = []
        for i in range(int(GetVarCount(self, varType))):
            names.append(GetVarName(self, i+1, varType))
        return names

    #docvariables
    def NeuronNames(self) -> List[str]:
        """Returns the list of neuron names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('neuron')

    #docvariables
    def EventNames(self) -> List[str]:
        """Returns the list of event names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('event')

    #docvariables
    def IntervalNames(self) -> List[str]:
        """Returns the list of interval variable names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('interval')

    #docvariables
    def WaveNames(self) -> List[str]:
        """Returns the list of waveform variable names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('wave')

    #docvariables
    def MarkerNames(self) -> List[str]:
        """Returns the list of marker variable names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('marker')

    #docvariables
    def ContinuousNames(self) -> List[str]:
        """Returns the list of continuous variable names in the document."""
        self._CheckIfValidDoc()
        return self._VarNames('continuous')
    
    def _VarsList(self, varType:str) -> List[NexVar]:
        """Internal helper function. Returns the list of variables."""
        theVars = []
        for i in range(int(GetVarCount(self, varType))):
            theVars.append(GetVar(self, i+1, varType))
        return theVars

    #docvariables
    def NeuronVars(self) -> List[NexVar]:
        """Returns the list of neuron variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('neuron')

    #docvariables
    def EventVars(self) -> List[NexVar]:
        """Returns the list of event variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('event')

    #docvariables
    def IntervalVars(self) -> List[NexVar]:
        """Returns the list of interval variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('interval')

    #docvariables
    def WaveVars(self) -> List[NexVar]:
        """Returns the list of waveform variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('wave')

    #docvariables
    def MarkerVars(self) -> List[NexVar]:
        """Returns the list of marker variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('marker')

    #docvariables
    def ContinuousVars(self) -> List[NexVar]:
        """Returns the list of continuous variables in the document."""
        self._CheckIfValidDoc()
        return self._VarsList('continuous')

    def GetRecordingStartTimeString(self) -> str:
        """Returns recording start time (if available) as a string in ISO 8601 format. 
        Use datetime.strptime(sts, '%Y-%m-%dT%H:%M:%S.%f') to convert to Python datetime object."""
        return self._Get("RecordingStartTime")

    def SetRecordingStartTime(self, dateTimeString:str):
        """Sets recording start time. dateTimeString should be in ISO 8601 format (use myDateTime.isoformat())""" 
        return self._SetProperty('RecordingStartTime', dateTimeString)


def _NexBuildFunctionCommandDict(functionName, pars):
    """Internal helper function. Packs function name and parameters into dictionary."""
    commandDict = {"type":"ExecuteNexScriptCommand"}
    commandDict["functionName"] = functionName
    commandDict["parameters"] = []
    for key, value in pars.items():
        # pass doc and var specs as inputs to function
        if isinstance(value, NexDoc ):
            commandDict["parameters"].append({ key : value.spec })
        elif isinstance( value, NexVar ):
            commandDict["parameters"].append( { key : value.spec } )
        else:
            commandDict["parameters"].append( {key : value} )
    return commandDict


def NexRun(functionName, pars):
    """Internal helper function. Runs specified function via nexLink.nexRunCommand."""
    commandDict = _NexBuildFunctionCommandDict(functionName, pars)
    # we are actually receiving utf-8 encoded bytes
    resultJsonBytes = nexLink.nexRunCommand(json.dumps(commandDict))
    if resultJsonBytes.startswith(b'error:'):
        raise RuntimeError(resultJsonBytes[6:].decode('utf-8'))
    returnedObject = json.loads(resultJsonBytes.decode('utf-8'))
    if not isinstance(returnedObject, dict):
        return returnedObject
    if 'type' in returnedObject:
        if returnedObject['type'] == 'number':
            return returnedObject['value']
        if returnedObject['type'] == 'string':
            return returnedObject['value']
        if returnedObject['type'] == 'docRef':
            return NexDoc(returnedObject)         
        if returnedObject['type'] == 'varRef':
            return NexVar(returnedObject)         

    return returnedObject

#run
def SetAppProperty(propertyName:str, propertyValue):
    """Sets application property."""
    return NexRun("SetAppProperty", locals())

#run
def UsePython3():
    """Function does nothing. Left for compatibility with internal NeuroExplorer scripting."""
    return

#run
def UsePython2():
    """Function does nothing. Left for compatibility with internal NeuroExplorer scripting."""
    return

#fileselection
def SelectFiles(initialDirectory:str='', extension:str='') -> List[str]:
    """Selects multiple files using Open File dialog. Returns a list of file path names."""
    command = {'type': 'SelectFiles', 'initialDirectory': initialDirectory, 'extension': extension}
    resultJsonBytes = nexLink.nexRunCommand(json.dumps(command))
    if resultJsonBytes.startswith(b'error:'):
        raise RuntimeError(resultJsonBytes[6:].decode('utf-8'))
    returnedObject = json.loads(resultJsonBytes.decode('utf-8'))
    if 'value' in returnedObject:
        return returnedObject['value']
    else:
       raise RuntimeError('invalid return value')

#run
def GetAppProperty(propertyName:str):
    """
    Returns specified application property.
    See https://www.neuroexplorer.com/docs/reference/scripting/run/GetAppProperty.html
    """
    command = {'type': 'GetAppProperty', 'propertyName': propertyName}
    resultJsonBytes = nexLink.nexRunCommand(json.dumps(command))
    if resultJsonBytes.startswith(b'error:'):
        raise RuntimeError(resultJsonBytes[6:].decode('utf-8'))
    returnedObject = json.loads(resultJsonBytes.decode('utf-8'))
    if not isinstance(returnedObject, dict):
        return returnedObject
    if 'type' in returnedObject:
        if returnedObject['type'] == 'number' or returnedObject['type'] == 'string':
            return returnedObject['value']
    return returnedObject


#ui
def Dialog(*arg) -> int:
    """
    Shows a dialog that can be used to specify the script parameters.
    Returns 0 if user pressed Cancel button, otherwise, returns 1.
    See https://www.neuroexplorer.com/docs/reference/scripting/ui/Dialog.html
    """
    command = {}
    command['type'] = 'Dialog'
    if isinstance( arg[0], NexDoc ):
        command['doc'] = arg[0].docId
    else:
        command['doc'] = arg[0]
    command['parameters'] = []
    numPars = (len(arg)-1)//3
    for i in range(numPars):
        par = {}
        par['value'] = arg[i*3+1]
        par['prompt'] = arg[i*3+2]
        par['type'] = arg[i*3+3]
        command['parameters'].append(par)
    
    resultJsonString = nexLink.nexRunCommand(json.dumps(command))
    returnedObject = json.loads(resultJsonString)
    if not isinstance(returnedObject, dict):
        return returnedObject
    wrapper = arg[len(arg) -1]
    if 'values' in returnedObject:
        values = returnedObject['values']
        if isinstance(values, list):
            for value in values:
                wrapper.append(value)

    if 'result' in returnedObject:
        return returnedObject['result']
    else:
        return 0

# functions that do not use NexRun (implemented separately for speed)   

#modvardata
def AddContValue(var:NexVar, timestamp:float, value:float):
    """Adds a new data point to the specified continuous variable. Timestamp is in seconds, value in milliVolts."""
    nexLink.nexAddContValue(var.varId, timestamp, value) 

#modvardata
def AddInterval(var:NexVar, interval_start:float, interval_end:float):
    """Adds a new interval to the specified interval variable. interval_start and interval_end are in seconds."""
    nexLink.nexAddInterval(var.varId, interval_start, interval_end) 

#modvardata
def AddTimestamp(var:NexVar, timestamp:float):
    """Adds a new timestamp (in seconds) to the specified event or neuron variable."""
    nexLink.nexAddTimestamp(var.varId, timestamp) 

def _JsonPlusNumList(jsonString:str, numList:List[float]):
    """Internal helper function. Runs generic command with numeric array and a JSON string with function parameters."""
    nexLink.nexJsonStringAndNumArray(jsonString, numList)

#anres
def GetNumResNCols(doc:NexDoc) -> int:
    """Returns the number of columns of the Numerical Results Window of the first graphical view of the document."""
    return int(doc.GetNumResNCols())

#anres
def GetNumResNRows(doc:NexDoc) -> int:
    """Returns the number of rows of the Numerical Results Window of the first graphical view of the document."""
    return int(doc.GetNumResNRows())

#anres
def GetNumRes(doc:NexDoc, row:int, col:int) -> float:
    """Returns the value of the specified cell in the Numerical Results Window of the first graphical view of the document."""
    return doc.GetNumResValue(row, col)

#anres
def GetNumResSummaryNCols(doc:NexDoc):
    """Returns the number of columns of the Numerical Results Summary Window of the first graphical view of the document."""
    return int(doc.GetNumResSummaryNCols())

#anres
def GetNumResSummaryNRows(doc:NexDoc):
    """Returns the number of rows of the Numerical Results Summary Window of the first graphical view of the document."""
    return int(doc.GetNumResSummaryNRows())

def PrintColumns(d):
    """ Prints lists of lists, for example, summary of numerical results."""
    # assuming vectors are the same length
    for i in range(len(d[0])):
        for j in range(len(d)):
            print(d[j][i]),
        print ('')
     
# end of custom functions   


#varoperations
def AbsOfContVar(doc:NexDoc, resultName:str, contVar:NexVar):
    """Calculates absolute value of the signal of a continuous variable."""
    return NexRun("AbsOfContVar", locals())

#analysis
def ApplyTemplate(doc:NexDoc, templateName:str):
    """Runs the analysis specified in the analysis template."""
    return NexRun("ApplyTemplate", locals())

#analysis
def ApplyTemplateToWindow(doc:NexDoc, templatename:str, windownumber:int):
    """Runs the analysis specified in the template and shows the result in the specified Graph window."""
    return NexRun("ApplyTemplateToWindow", locals())

#math
def BitwiseAnd(value1:int, value2:int) -> int:
    """Returns the result of the bitwise AND operation."""
    return int(int(value1) & int(value2))

#math
def BitwiseOr(value1:int, value2:int) -> int:
    """Returns the result of the bitwise OR operation."""
    return int(int(value1) | int(value2))

#str
def CharToNum(oneCharString:str) -> int:
    """Converts a one-character string to a number (a character's ASCII code)."""
    return ord(oneCharString)

#filereadandwrite
def CloseDocument(doc:NexDoc):
    """Closes the specified document."""
    return NexRun("CloseDocument", locals())

#excel
def CloseExcelFile(filePath:str):
    """Closes the specified Excel file if the file is open."""
    return NexRun("CloseExcelFile", locals())

#readwritebinaryandtext
def CloseFile(fileID):
    """Closes the specified file."""
    return NexRun("CloseFile", locals())

#ppt
def ClosePowerPointFile(filePath:str):
    """Closes the specified PowerPoint file if the file is open."""
    return NexRun("ClosePowerPointFile", locals())

#docvariables
def ContVarStoreValuesAsFloats(contVar:NexVar):
    """Converts storage of continuous values from 16-bit integers to 32-bit floats."""
    return NexRun("ContVarStoreValuesAsFloats", locals())

#varoperations
def ContAdd(contVar:NexVar, numberToAdd:float) -> NexVar:
    """Creates a new continuous variable with values contVar[i]+numberToAdd, returns a reference to the new variable."""
    return NexRun("ContAdd", locals())

#varoperations
def ContMult(contVar:NexVar, numberToMultiply:float) -> NexVar:
    """Creates a new continuous variable with values contVar[i]*numberToMultiply, returns a reference to the new variable."""
    return NexRun("ContMult", locals())

#varoperations
def ContAddCont(contVar1:NexVar, contVar2:NexVar) -> NexVar:
    """Creates a new continuous variable with values contVar1[i]+contVar2[i], returns a reference to the new variable."""
    return NexRun("ContAddCont", locals())

#varoperations
def ContSubtractCont(contVar1:NexVar, contVar2:NexVar) -> NexVar:
    """Creates a new continuous variable with values contVar1[i]-contVar2[i], returns a reference to the new variable."""
    return NexRun("ContSubtractCont", locals())

#varoperations
def ContMultCont(contVar1:NexVar, contVar2:NexVar) -> NexVar:
    """Creates a new continuous variable with values contVar1[i]*contVar2[i], returns a reference to the new variable."""
    return NexRun("ContMultCont", locals())

#varoperations
def ContDivCont(contVar1:NexVar, contVar2:NexVar) -> NexVar:
    """Creates a new continuous variable with values contVar1[i]/contVar2[i], returns a reference to the new variable."""
    return NexRun("ContDivCont", locals())

#copyingvars
def CopySelectedVarsToAnotherFile(fromDoc:NexDoc, toDoc:NexDoc):
    """Copies selected variables from one file to another."""
    return NexRun("CopySelectedVarsToAnotherFile", locals())

#varoperations
def DecimateContVar(doc:NexDoc, resultName:str, contVar:NexVar, decimationFactor:int):
    """Decimates specified continuous variable."""
    return NexRun("DecimateContVar", locals())

#deletingvars
def Delete(doc:NexDoc, var:NexVar):
    """Deletes the specified variable from the file."""
    doc._OnDeleteVar(var)
    return NexRun("Delete", locals())

#deletingvars
def DeleteVar(doc:NexDoc, number:int, varType:str):
    """Deletes the specified variable from the file."""
    doc._OnDeleteVar(GetVar(doc, number, varType))
    return NexRun("DeleteVar", locals())

#varselection
def Deselect(doc:NexDoc, var:NexVar):
    """Deselects the specified variable."""
    return NexRun("Deselect", locals())

#varselection
def DeselectAll(doc:NexDoc):
    """Deselects all variables"""
    return NexRun("DeselectAll", locals())

#varselection
def DeselectVar(doc:NexDoc, number:int, varType:str):
    """Deselects the specified variable."""
    return NexRun("DeselectVar", locals())

#analysis
def DisableRecalcOnSelChange():
    """Disables recalculation of analyses when the list of selected variables changes."""
    return NexRun("DisableRecalcOnSelChange", locals())

#varselection
def EnableRecalcOnSelChange():
    """Enables recalculation of analyses when the list of selected variables changes."""
    return NexRun("EnableRecalcOnSelChange", locals())

#varoperations
def EndOfInterval(intervalVar) -> NexVar:
    """Creates the new event based on the specified interval variable. Copies the end of each interval of the interval variable to the result."""
    return NexRun("EndOfInterval", locals())

#matlab
def ExecuteMatlabCommand(command:str):
    """Sends the string command to Matlab and executes the command in Matlab."""
    return NexRun("ExecuteMatlabCommand", locals())

#readwritebinaryandtext
def FileSeek(fileID, offset:int, type:str):
    """Repositions file pointer by the specified offset."""
    return NexRun("FileSeek", locals())

#varoperations
def FilterContinuousVariable(doc:NexDoc, contVar:NexVar, filteredVarName:str, filterType:str, filterOrder:int, freq1:float, freq2:float):
    """Filters the specified continuous variable using the specified frequency filter."""
    return NexRun("FilterContinuousVariable", locals())

#str
def Find(string1:str, string2:str):
    """Looks for a substring inside a specified string."""
    return NexRun("Find", locals())

#varoperations
def FirstAfter(var1:NexVar, var2:NexVar, fromTime:float, toTime:float) -> NexVar:
    """Creates the new event containing the first timestamp of var1 in each of the intervals [var2+fromTime, var2+toTime]."""
    return NexRun("FirstAfter", locals())

#varoperations
def FirstInInterval(var:NexVar, intervalVar:NexVar) -> NexVar:
    """Creates the new event. For each interval of the specified interval variable, the first timestamp in this interval is copied to the result."""
    return NexRun("FirstInInterval", locals())

#varoperations
def FirstNAfter(var1:NexVar, var2:NexVar, count:int) -> NexVar:
    """Creates the new event containing the first N timestamps of one variable after each of the timestamps of the second variable."""
    return NexRun("FirstNAfter", locals())


#filereadandwrite
def GetActiveDocument() -> NexDoc:
    """Returns a reference to the currently active document."""
    return NexRun("GetActiveDocument", locals())

#math
def GetBinCount(var:NexVar, timeMin:float, timeMax:float):
    """Calculates the number of timestamps in the specified time range."""
    return int(NexRun("GetBinCount", locals()))

#math
def GetBit(x:int, oneBasedBitNumber:int):
    """Returns the value of the specified bit (1 to 32)."""
    if oneBasedBitNumber < 1 or oneBasedBitNumber > 32:
        raise ValueError('invalid bit number')
    return (int(x) >> (oneBasedBitNumber-1)) & 1 

#varproperties
def GetContNumDataPoints(var:NexVar) -> int:
    """Returns the number of data points in the continuous variable."""
    return int(NexRun("GetContNumDataPoints", locals()))

#matlab
def GetContVarFromMatlab(doc:NexDoc, MatrixName:str, TimestampOfFirstValue:float, TimeStep:float):
    """Imports the specified matrix from Matlab. Each column of the matrix is imported as a continuous variable."""
    return NexRun("GetContVarFromMatlab", locals())

#matlab
def GetContVarWithTimestampsFromMatlab(doc:NexDoc, MatrixName:str, UseFirstDeltaAsDigRate:int):
    """Imports the specified 2-column matrix containing continuous variable data from Matlab."""
    return NexRun("GetContVarWithTimestampsFromMatlab", locals())

#docproperties
def GetDocComment(doc:NexDoc) -> str:
    """Returns the document comment string."""
    return NexRun("GetDocComment", locals())

#docproperties
def GetDocEndTime(doc:NexDoc) -> float:
    """Returns the maximum timestamp value (in seconds) for all the document variables."""
    return NexRun("GetDocEndTime", locals())

#docproperties
def GetDocStartTime(doc:NexDoc) -> float:
    """Returns the minimum timestamp value (in seconds) for all the document variables."""
    return NexRun("GetDocStartTime", locals())

#docproperties
def GetDocPath(doc:NexDoc) -> str:
    """Returns the full path of the data file."""
    return NexRun("GetDocPath", locals())

#docproperties
def GetDocTitle(doc:NexDoc) -> str:
    """Returns the data file name."""
    return NexRun("GetDocTitle", locals())

#str
def GetField(stringWithFields, fieldnumber) -> str:
    """Returns the string field."""
    return NexRun("GetField", locals())

#fileselection
def GetFileCount(fileFilter:str) -> int:
    """Returns the number of files that match the file filter."""
    return int(NexRun("GetFileCount", locals()))

#fileselection
def GetFileName(index:int) -> str:
    """Returns the file name for the specified index after GetFileCount() was called."""
    return NexRun("GetFileName", locals())

#math
def GetFirstGE(var:NexVar, time:float) -> int:
    """Returns the index of the first timestamp in the specified variable that is greater than or equal to the specified number."""
    return int(NexRun("GetFirstGE", locals()))

#math
def GetFirstGT(var:NexVar, time:float) -> int:
    """Returns the index of the first timestamp in the specified variable that is greater than the specified number."""
    return int(NexRun("GetFirstGT", locals()))

#matlab
def GetIntervalVarFromMatlab(doc:NexDoc, MatrixName:str):
    """Imports the specified matrix containing intervals from Matlab."""
    return NexRun("GetIntervalVarFromMatlab", locals())

#varproperties
def GetName(var:NexVar) -> str:
    """Returns the name of the variable."""
    return NexRun("GetName", locals())

#str
def GetNumFields(stringWithFields:str) -> int:
    """Returns the number of fields in the string. The field is a substring that does not contain spaces, tabs or commas."""
    return int(NexRun("GetNumFields", locals()))

#anres
def GetNumResColumnName(doc:NexDoc, col:int) -> str:
    """Returns the name of the specified column of the Numerical Results Window."""
    return NexRun("GetNumResColumnName", locals())

#anres
def GetNumResSummaryColumnName(doc:NexDoc, col:int) -> str:
    """Returns the name of the specified column of the Numerical Results Summary Window of the first graphical view of the document."""
    return NexRun("GetNumResSummaryColumnName", locals())

#anres
def GetNumResSummaryData(doc:NexDoc, row:int, col:int) -> str:
    """Returns the string value of the specified cell in the Numerical Results Summary Window of the first graphical view of the document."""
    return NexRun("GetNumResSummaryData", locals())

#varproperties
def GetSpikeCount(var:NexVar) -> int:
    """Returns the number of timestamps in the variable."""
    return int(NexRun("GetSpikeCount", locals()))

#analysis
def GetTemplateParValue(templateName:str, paramName:str) -> str:
    """Returns the value of the specified template parameter as string. Python only"""
    return NexRun("GetTemplateParValue", locals())

#docproperties
def GetTimestampFrequency(doc:NexDoc) -> float:
    """Returns the frequency used in the internal representation of the timestamps."""
    return NexRun("GetTimestampFrequency", locals())

#docvariables
def GetVar(doc:NexDoc, varNumber:int, varType:str) -> NexVar:
    """Returns the reference to the specified variable."""
    return NexRun("GetVar", locals())

#docvariables
def GetVarByName(doc:NexDoc, name:str) -> NexVar:
    """Returns the reference to the variable which has the specified name."""
    return NexRun("GetVarByName", locals())

#docvariables
def GetVarCount(doc:NexDoc, varType:str) -> int:
    """Returns the number of variables of the specified type in the file."""
    return int(NexRun("GetVarCount", locals()))

#matlab
def GetVarFromMatlab(doc:NexDoc, varname:str, isneuron:int):
    """Imports the specified Matlab timestamps vector as neuron or event variable."""
    return NexRun("GetVarFromMatlab", locals())

#docvariables
def GetVarName(doc:NexDoc, varNumber:int, varType:str) -> str:
    """Returns the name of the specified variable."""
    return NexRun("GetVarName", locals())

#varproperties
def GetVarSpikeCount(doc:NexDoc, varNumber:int, varType:str) -> str:
    """Returns the number of timestamps in the specified variable."""
    return int(NexRun("GetVarSpikeCount", locals()))

#varoperations
def IntAnd(intervalVar1:NexVar, intervalVar2:NexVar) -> NexVar:
    """Creates a new Interval Variable that contains intersections of the intervals of intervalVar1 and intervalVar2."""
    return NexRun("IntAnd", locals())

#varoperations
def IntervalFilter(var:NexVar, intervalVar:NexVar) -> NexVar:
    """Creates the new event containing all the timestamps of the specified event or neuron variable that are in the intervals of the specified interval variable."""
    return NexRun("IntervalFilter", locals())

#varoperations
def IntFind(intervalVar:NexVar, eventVar:NexVar) -> NexVar:
    """Finds all intervals that contain at least one timestamp of the specified event or neuron variable."""
    return NexRun("IntFind", locals())

#varoperations
def IntOpposite(doc:NexDoc, intervalVariable:NexVar) -> NexVar:
    """Creates a new interval variable that contains intervals 'complementary' to the intervals of the specified interval variable."""
    return NexRun("IntOpposite", locals())

#varoperations
def IntOr(doc:NexDoc, intervalVar1:NexVar, intervalVar2:NexVar) -> NexVar:
    """Creates a new Interval Variable that contains unions of the intervals of intervalVar1 and intervalVar2."""
    return NexRun("IntOr", locals())

#varoperations
def IntSize(intervalVar:NexVar, minInt:float, maxInt:float) -> NexVar:
    """Creates a new Interval Variable that contains the intervals with the specified length range."""
    return NexRun("IntSize", locals())

#varoperations
def ISIFilter(var:NexVar, minISI:float) -> NexVar:
    """Creates the new event containing the timestamps of the specified variable that have preceding interspike intervals larger than the specified value."""
    return NexRun("ISIFilter", locals())

#varselection
def IsSelected(var:NexVar) -> int:
    """Returns 1 if the specified variable is selected, 0 otherwise."""
    return int(NexRun("IsSelected", locals()))

#varoperations
def Join(var1:NexVar, var2:NexVar) -> NexVar:
    """Creates the new event that contains the timestamps of the two specified variables."""
    return NexRun("Join", locals())

#varoperations
def LastBefore(var1:NexVar, var2:NexVar, fromTime:float, toTime:float) -> NexVar:
    """Creates the new event containing the last timestamp of var1 in each of the intervals [var2+fromTime, var2+toTime]."""
    return NexRun("LastBefore", locals())

#varoperations
def LastInInterval(var:NexVar, intervalVar:NexVar) -> NexVar:
    """Creates the new event. For each interval of the specified interval variable, the last timestamp in this interval is copied to the result."""
    return NexRun("LastInInterval", locals())

#str
def Left(string:str, nchar:int) -> str:
    """Returns a substring that starts at the beginning of the string."""
    return NexRun("Left", locals())

#varoperations
def LinearCombinationOfContVars(doc:NexDoc, resultName:str, contVar1:NexVar, coeff1:float, contVar2:NexVar, coeff2:float):
    """Calculates a linear combination of two continuous variables."""
    return NexRun("LinearCombinationOfContVars", locals())

#varoperations
def MakeIntervals(var:NexVar, shiftMin:float, shiftMax:float) -> NexVar:
    """Creates new interval variable with intervals [varTimestamp+shiftmin, varTimestamp+shiftMax]."""
    return NexRun("MakeIntervals", locals())

#varoperations
def MakeIntFromEnd(intStartVar:NexVar, intEndVar:NexVar, shift1:float, shift2:float) -> NexVar:
    """Creates new interval variable. For each timestamp tend of the intEndVar, it looks for the last timestamp (tstart) of the intStartVar before tend. If tstart is after the previous timestamp of intEndVar, it adds the interval [tstart+shift1, tend+shift2] to the result."""
    return NexRun("MakeIntFromEnd", locals())

#varoperations
def MakeIntFromStart(intStartVar:NexVar, intEndVar:NexVar, shift1:float, shift2:float) -> NexVar:
    """Creates new interval variable. For each timestamp tstart of intStartVar, it looks for the first timestamp tend of the intEndVar after tstart. If tend is before the next timestamp of intStartVar, it adds the interval [tstart+shift1, tend+shift2] to the result."""
    return NexRun("MakeIntFromStart", locals())

#varoperations
def MarkerExtract(doc:NexDoc, MarkerVariableName:str, ExtractString:str) -> NexVar:
    """Creates a new event variable based on existing marker variable."""
    return NexRun("MarkerExtract", locals())

#filereadandwrite
def MergeFiles(name_list:List[str], gap:float) -> NexDoc:
    """Opens and merges the specified files, returns the reference to the merged file."""
    return NexRun("MergeFiles", locals())

#str
def Mid(string:str, nstartchar:int, nchar:int) -> str:
    """Returns the specified substring."""
    return NexRun("Mid", locals())

#analysis
def ModifyTemplate(doc:NexDoc, templateName:str, paramName:str, newParValue:str):
    """Modifies one of the template parameters."""
    return NexRun("ModifyTemplate", locals())

#creatingnewvars
def NewContVar(doc:NexDoc, frequency:float, mVmin:float, mVmax:float) -> NexVar:
    """Creates a new continuous variable."""
    return NexRun("NewContVar", locals())

#creatingnewvars
def NewContVarWithFloats(doc:NexDoc, frequency:float) -> NexVar:
    """Creates a new continuous variable, returns a reference to the new variable."""
    return NexRun("NewContVarWithFloats", locals())

#filereadandwrite
def NewDocument(frequency:float) -> NexDoc:
    """Creates a new document (data file) with the specified timestamp frequency."""
    return NexRun("NewDocument", locals())

#creatingnewvars
def NewEvent(doc:NexDoc, count:int) -> NexVar:
    """Creates a new timestamped variable."""
    return NexRun("NewEvent", locals())

#creatingnewvars
def NewIntEvent(doc:NexDoc, count:int = 0) -> NexVar:
    """Creates a new interval variable."""
    return NexRun("NewIntEvent", locals())

#creatingnewvars
def NewPopVector(doc:NexDoc, type:int) -> NexVar:
    """Creates a new population vector."""
    return NexRun("NewPopVector", locals())

#varoperations
def NotSync(var1:NexVar, var2:NexVar, fromTime:float, toTime:float) -> NexVar:
    """Creates the new event containing all the timestamps of var1 that are NOT in the intervals [var2+fromTime, var2+toTime]."""
    return NexRun("NotSync", locals())

#varoperations
def NthAfter(var1:NexVar, var2:NexVar, N:int) -> NexVar:
    """Creates the new variable with the N-th timestamp in var1 after each timestamp in var2."""
    return NexRun("NthAfter", locals())

#str
def NumToChar(number:int) -> str:
    """Converts a number to a one-character string containing a character with the ASCII code equal to the number."""
    return chr(number)

#str
def NumToStr(number:float, formatString:str = '') -> str:
    """Converts number to string using an optional format string."""
    if formatString == '':
        return str(float(number))
    else:
        if not formatString[0] == '%':
            raise ValueError('invalid format string (first character should be %)')
        fmt = '{:'+formatString[1:]+'}'
        return fmt.format(float(number))

#filereadandwrite
def OpenDocument(filePath:str) -> NexDoc:
    """Opens a data file with the specified path. Returns a reference to the opened document."""
    return NexRun("OpenDocument", locals())

#readwritebinaryandtext
def OpenFile(filePath:str, mode:str) -> int:
    """Opens text file using the specified mode, returns file ID."""
    return int(NexRun("OpenFile", locals()))

#varoperations
def PositionSpeed(varX:NexVar, varY:NexVar, deltaT:float, smoothRadius:float) -> NexVar:
    """Calculates the position speed from X and Y coordinate variables and creates a new continuous variable."""
    return NexRun("PositionSpeed", locals())

#analysis
def PrintGraphics(doc:NexDoc):
    """Prints the contents of the first graphical window of the document."""
    return NexRun("PrintGraphics", locals())

#readwritebinaryandtext
def ReadBinary(fileID:int, valueType:str) -> float:
    """Reads a binary value of a specified type from a file."""
    return NexRun("ReadBinary", locals())

#analysis
def RecalculateAnalysisInWindow(doc:NexDoc, graphWindowNumber:int):
    """Forces recalculation of analysis in the specified graphic window."""
    return NexRun("RecalculateAnalysisInWindow", locals())

#varoperations
def Rename(doc:NexDoc, var:NexVar, newName:str):
    """Renames the specified variable."""
    doc._OnRenameVar(var, newName)
    return NexRun("Rename", locals())

#str
def Right(string:str, nchar:int) -> str:
    """Returns a substring that ends at the end of the string."""
    return NexRun("Right", locals())

#math
def RoundToTS(doc:NexDoc, time:float) -> float:
    """Rounds the specified number to the nearest timestamp value."""
    return NexRun("RoundToTS", locals())

#filereadandwrite
def SaveAsTextFile(doc:NexDoc, filePath:str):
    """Saves the document in the text file with the specified file name."""
    return NexRun("SaveAsTextFile", locals())

#filereadandwrite
def SaveDocument(doc:NexDoc):
    """Saves the specified document."""
    return NexRun("SaveDocument", locals())

#filereadandwrite
def SaveDocumentAs(doc:NexDoc, filePath:str):
    """Saves the specified document (in .nex, .nex5 or .edf format) in a file with the specified file path."""
    return NexRun("SaveDocumentAs", locals())

#analysis
def SaveGraphics(doc:NexDoc, filePath:str, asPng:int):
    """Saves the graphics of the first graphics window of the document to a WMF, PNG or SVG file."""
    return NexRun("SaveGraphics", locals())

#analysis
def SaveNumResults(doc:NexDoc, fileName:str):
    """Saves the numerical results to a text file with the specified name."""
    return NexRun("SaveNumResults", locals())

#analysis
def SaveNumSummary(doc:NexDoc, filename:str):
    """Saves the summary of numerical results to a text file with the specified name."""
    return NexRun("SaveNumSummary", locals())

#varselection
def Select(doc:NexDoc, var:NexVar):
    """Selects the specified variable for analysis."""
    return NexRun("Select", locals())

#varselection
def SelectAll(doc:NexDoc):
    """Selects all the variables for analysis."""
    return NexRun("SelectAll", locals())

#varselection
def SelectAllEvents(doc:NexDoc):
    """Selects all event type variables for analysis."""
    return NexRun("SelectAllEvents", locals())

#varselection
def SelectAllNeurons(doc:NexDoc):
    """Selects all neuron type variables for analysis."""
    return NexRun("SelectAllNeurons", locals())

#varoperations
def SelectEven(var:NexVar) -> NexVar:
    """Creates the new event containing even (2nd, 4th, etc.) timestamps of the specified variable."""
    return NexRun("SelectEven", locals())

#fileselection
def SelectFile() -> str:
    """Opens File Open dialog and returns the path of the file selected in the dialog."""
    return NexRun("SelectFile", locals())

#varoperations
def SelectOdd(var:NexVar) -> NexVar:
    """Creates the new event containing the odd (1st, 3rd, etc.) timestamps of the specified variable."""
    return NexRun("SelectOdd", locals())

#varoperations
def SelectRandom(var:NexVar, nSelect:int) -> NexVar:
    """Creates the new event containing randomly selected timestamps of the specified variable."""
    return NexRun("SelectRandom", locals())

#varoperations
def SelectTrials(var:NexVar, selectList:str) -> NexVar:
    """Creates the new event containing the specified timestamps of a variable."""
    return NexRun("SelectTrials", locals())

#varselection
def SelectVar(doc:NexDoc, number:int, varType:str):
    """Selects the specified variable for analysis."""
    return NexRun("SelectVar", locals())

#analysis
def SendGraphicsToPowerPoint(doc:NexDoc, presentationPath:str, slideTitle:str, comment:str, addParameters:int, useBitmap:int):
    """Sends the contents of the first graphical window of the document to the specified PowerPoint presentation."""
    return NexRun("SendGraphicsToPowerPoint", locals())

#anres
def SendResultsSummaryToExcel(doc:NexDoc, fileName:str, worksheetName:str, useFirstEmptyRow:int, cellName:str, includeHeader:int, includeFileName:int):
    """Sends summary of numerical results (of the first graphics window of the document) to Excel."""
    return NexRun("SendResultsSummaryToExcel", locals())

#anres
def SendResultsToExcel(doc:NexDoc, fileName:str, worksheetName:str, useFirstEmptyRow:int, cellName:str, includeHeader:int, includeFileName:int):
    """Sends numerical results (of the first graphics window of the document) to Excel."""
    return NexRun("SendResultsToExcel", locals())

#matlab
def SendSelectedVarsToMatlab(doc:NexDoc):
    """Sends selected variables to Matlab."""
    return NexRun("SendSelectedVarsToMatlab", locals())

#docproperties
def SetDocEndTime(doc:NexDoc, endtime:float):
    """Sets the length (in seconds) of experimental session for the document."""
    return NexRun("SetDocEndTime", locals())

#docproperties
def SetDocStartTime(doc:NexDoc, start_time:float):
    """Sets the start (in seconds) of experimental session for the document."""
    return NexRun("SetDocStartTime", locals())

#excel
def SetExcelCell(worksheet:str, cell:str, text:str, excelFilePath:str = ''):
    """Sets the text value of the specified cell in Excel."""
    return NexRun("SetExcelCell", locals())

#docvariables
def SetNeuronType(doc:NexDoc, var:NexVar, type:int):
    """Changes the type of the specified timestamped variable."""
    return NexRun("SetNeuronType", locals())

#varoperations
def Shift(var:NexVar, shiftBy:float) -> NexVar:
    """Returns a new variable with all the timestamps of variable var shifted in time by shiftBy seconds."""
    return NexRun("Shift", locals())

#run
def Sleep(nms:float):
    """Pauses execution of the script for nms milliseconds."""
    return NexRun("Sleep", locals())

#varoperations
def StartOfInterval(intervalVar:NexVar) -> NexVar:
    """Creates the new event. Copies the start of each interval of the specified interval variable to the result."""
    return NexRun("StartOfInterval", locals())

#str
def StrLength(stringVal:str) -> int:
    """Calculates the number of characters in the string."""
    return int(NexRun("StrLength", locals()))

#str
def StrToNum(stringRepresentingNumber:str) -> float:
    """Converts string to number."""
    return float(stringRepresentingNumber)

#varoperations
def Sync(var1:NexVar, var2:NexVar, fromTime:float, toTime:float) -> NexVar:
    """Creates the new event containing all the timestamps of var1 that are in the intervals [var2+fromTime, var2+toTime]."""
    return NexRun("Sync", locals())

#readwritebinaryandtext
def WriteLine(fileIDI:int, lineString:str):
    """Writes a line of text to a text file."""
    return NexRun("WriteLine", locals())

#ui
def ExecuteMenuCommand(menuName:str):
    """Executes menu command."""
    return NexRun("ExecuteMenuCommand", locals())

#analysis
def SaveResults(doc:NexDoc, fileName:str, comment:str):
    """ Saves numerical and graphical results as well as an analysis template for the first graphical view of the document."""
    return NexRun("SaveResults", locals())

#resultfilesreadandwrite
def OpenSavedResult(filePath:str):
    """Opens saved result file."""
    return NexRun("OpenSavedResult", locals())

#run
def RunScript(filePath:str):
    """Runs NexScript script."""
    if not os.path.splitext(filePath)[1].lower() == '.py':
        return NexRun("RunScript", locals())
    else: 
        raise ValueError('to run Python script, use import statement')

#analysis
def SelectVariablesIn1DView(doc:NexDoc, varNamesInCommaSeparatedString:str):
    """Selects variables in 1D View using provided comma separated list of variable names."""
    return NexRun("SelectVariablesIn1DView", locals())

#varselection
def SelectVariables(doc:NexDoc, varNamesInCommaSeparatedString:str):
    """Selects variables for analysis using provided comma separated list of variable names. Python only"""
    DeselectAll(doc)
    # split on empty string returns a list with one empty string. we want to avoid that
    if varNamesInCommaSeparatedString == '':
        return
    s = StringIO(varNamesInCommaSeparatedString)
    varNamesList = list(csv.reader(s, skipinitialspace=True))[0]
    for varName in varNamesList:
        name = varName.strip()
        var = doc[name]
        if not var:
            raise ValueError('unable to find variable "{}" in the document'.format(name))
        Select(doc, var)

#docproperties
def SetDocTimestampFrequency(doc:NexDoc, freq:float):
    """Sets the document timestamp frequency"""
    return NexRun("SetDocTimestampFrequency", locals())


def GetFilePathWithoutExtension(thePath:str) -> str:
    """ Returns the file path without file extension."""
    return os.path.splitext(thePath)[0]


#ui
def ActivateWindow(docPath:str, windowName:str):
    """Activates the specified child window."""
    return NexRun("ActivateWindow", locals())

#ui
def CloseWindow(docPath:str, windowName:str):
    """Closes specified child window."""
    return NexRun("CloseWindow", locals())

#math
def PoissonSurprise(k:int, freq:float, time:float):
    """Returns Poisson Surprise -  the probability that ``k`` or more spikes occur randomly in a period of length ``time``. Python only"""
    return NexRun("PoissonSurprise", locals())


# FUNCTIONS THAT CALL PYTHON

#math
def seed(iseed:int):
    """Initializes random number generator."""
    random.seed(iseed)
    return

#math
def rand():
    """Returns a number with a uniform random distribution from 0 to 1."""
    return random.uniform(0,1)

#math
def expo(fmean:float):
    """Returns a random value exponentially distributed with the specified mean."""
    d = 0.0;
    while d == 0.0:
        d = random.uniform(0,1)
    return -math.log( d ) * fmean;

#math
def floor(theNumber:float):
    """Returns math.floor(theNumber)."""
    return math.floor(theNumber)

#math
def ceil(theNumber:float):
    """Returns math.ceil(theNumber)."""
    return math.ceil(theNumber)

#math
def round(theNumber:float):
    """Returns round(theNumber)."""
    return builtins.round(theNumber)

#math
def abs(theNumber:float):
    """Returns abs(theNumber)."""
    return builtins.abs(theNumber)

#math
def sqrt(theNumber:float):
    """Returns math.sqrt(theNumber)."""
    return math.sqrt(theNumber)

#math
def pow(theNumber:float, thePower:float):
    """Returns math.pow(theNumber, thePower)."""
    return math.pow(theNumber, thePower)

#math
def exp(theNumber:float):
    """Returns math.exp(theNumber)."""
    return math.exp(theNumber)

#math
def min(x:float, y:float):
    """Returns min(x, y)."""
    return builtins.min(x, y)

#math
def max(x:float, y:float):
    """Returns max(x, y)."""
    return builtins.max(x, y)

#math
def log(theNumber:float):
    """Returns math.log(theNumber)."""
    return math.log(theNumber)

#math
def sin(theNumber:float):
    """Returns math.sin(theNumber)."""
    return math.sin(theNumber)

#math
def cos(theNumber:float):
    """Returns math.cos(theNumber)."""
    return math.cos(theNumber)

#math
def tan(theNumber:float):
    """Returns math.tan(theNumber)."""
    return math.tan(theNumber)

#math
def acos(theNumber:float):
    """Returns math.acos(theNumber)."""
    return math.acos(theNumber)

#math
def asin(theNumber):
    """Returns math.asin(theNumber)."""
    return math.asin(theNumber)

#math
def atan(theNumber:float):
    """Returns math.atan(theNumber)."""
    return math.atan(theNumber)

# repeat doc member functions

def NeuronNames(doc:NexDoc) -> List[str]:
    """Returns the list of neuron names in the document."""
    return doc.NeuroNames()

def EventNames(doc:NexDoc) -> List[str]:
    """Returns the list of event names in the document."""
    return doc.EventNames()

def IntervalNames(doc:NexDoc) -> List[str]:
    """Returns the list of interval variable names in the document."""
    return doc.IntervalNames()

def WaveNames(doc:NexDoc) -> List[str]:
    """Returns the list of waveform variable names in the document."""
    return doc.WaveNames()

def MarkerNames(doc:NexDoc) -> List[str]:
    """Returns the list of marker variable names in the document."""
    return doc.MarkerNames()

def ContinuousNames(doc:NexDoc) -> List[str]:
    """Returns the list of continuous variable names in the document."""
    return doc.ContinuousNames()

def NeuronVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of neuron variables in the document."""
    return doc.NeuronVars()

def EventVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of event variables in the document."""
    return doc.EventVars()

def IntervalVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of interval variables in the document."""
    return doc.IntervalVars()

def WaveVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of wave variables in the document."""
    return doc.WaveVars()

def MarkerVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of marker variables in the document."""
    return doc.MarkerVars()

def ContinuousVars(doc:NexDoc) -> List[NexVar]:
    """Returns the list of continuous variables in the document."""
    return doc.ContinuousVars()

def GetAllNumericalResults(doc:NexDoc) -> List[List[float]]:
    """
    Returns the list of all values (as a list of lists) 
    in the Numerical Results Window of the first graphical view of the document.
    The first element contains the values of the first column of numerical results.
    The second contains values of the second column of numerical results, etc.
    """
    return doc.GetAllNumericalResults()

def GetAllNumResSummaryData(doc:NexDoc) -> List[List[str]]:
    """
    Returns all values of numerical results summary as a list of lists.
    The first element contains the values of the first column of numerical results summary.
    The second contains values of the second column of numerical results summary, etc.
    """
    return doc.GetAllNumResSummaryData()

def GetNumResSummaryColumnNames(doc:NexDoc) -> List[str]:
    """Returns the list of column names in the Summary of Numerical Results Window of the first graphical view of the document."""
    return doc.GetNumResSummaryColumnNames()

def GetNumResColumnNames(doc:NexDoc) -> List[str]:
    """Returns the list of column names in the Numerical Results window of the first graphical view of the document."""
    return doc.GetNumResColumnNames()

#run
def InstallPackageIfNeeded(packageName):
    """Function does nothing. Left for compatibility with internal NeuroExplorer scripting."""
    return

def FilterContinuousVariableEx(contVar:NexVar, filterType:str, filterImplementation:str, filterOrder:int, freq1:float, freq2:float, ripple:float=1.0):
    """Filters continuous variable using IIR of FIR frequency filter."""
    theLocals = locals()
    theLocals['contVar'] = contVar.varId
    command = theLocals
    command['type'] = 'FilterContinuousVariableEx'
    resultJsonBytes = nexLink.nexRunCommand(json.dumps(command))
    if resultJsonBytes.startswith(b'error:'):
        raise RuntimeError(resultJsonBytes[6:].decode('utf-8'))
    returnedObject = json.loads(resultJsonBytes.decode('utf-8'))
    if isinstance(returnedObject, dict):
        if 'type' in returnedObject:
            if returnedObject['type'] == 'varRef':
                return NexVar(returnedObject)         

    raise RuntimeError("invalid return from FilterContinuousVariableEx")


def DialogEx(jsonString:str) -> str:
    """ Shows dialog based on JSON string. 
        Returns JSON string representing a Python dictionary 
        with dialog result and parameter values.
    """
    return NexRun("DialogEx", locals())


def GetRecordingStartTimeString(doc:NexDoc) -> str:
    """Returns recording start time (if available) as a string in ISO 8601 format. 
    Use datetime.strptime(sts, '%Y-%m-%dT%H:%M:%S.%f') to convert to Python datetime object."""
    return doc.GetRecordingStartTimeString()


def SetRecordingStartTime(doc:NexDoc, dateTimeString:str):
    """Sets recording start time. dateTimeString should be in ISO 8601 format (use myDateTime.isoformat())""" 
    return doc.SetRecordingStartTime(dateTimeString)


def CloseNonDataWindows(self):
    """ Closes all analysis, 1D view or 3D view windows of the document."""
    return self._SetProperty('CloseAllWindowsExceptDataWindow', '')

