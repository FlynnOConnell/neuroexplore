import socket
import struct
import json
import array
import os
import tempfile
import base64


class SocketAdaptor:
    def __init__(self):
        self.isConnected = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(2)

        thePort = 50379
        # the port number should be saved in PythonPortNumber.txt by NeuroExplorer
        try:
            pythonPortNumberFilePath = "C:/ProgramData/Nex Technologies/NeuroExplorer 5 x64/PythonPortNumber.txt"
            with open(pythonPortNumberFilePath) as f:
                lines = f.readlines()
            if len(lines) > 0:
                thePort = int(lines[0].strip())
        except: 
            thePort = 50379

        self._socket.connect(('localhost', thePort))
        self._socket.settimeout(2000)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.isConnected = True

    def __del__(self):
        self._socket.close()

    def sendMessage(self, msg):
        # Prefix each message with a 8-byte length
        msg = struct.pack('<q', len(msg)) + msg.encode('utf-8')
        self._socket.sendall(msg)

    def receiveMessage(self):
        # Read message length and unpack it into an integer
        messageLengthBytes = self.receiveAll(8)
        if not messageLengthBytes:
            return None
        theLength = struct.unpack('<q', messageLengthBytes)[0]
        # Read the message data
        if theLength <= 0:
            raise RuntimeError('received invalid reply from NeuroExplorer')
        msg = self.receiveAll(theLength)
        if not msg:
            raise RuntimeError('received invalid reply from NeuroExplorer')
        return msg

    def receiveAll(self, n):
        data = bytes()
        while len(data) < n:
            count = n - len(data)
            packet = self._socket.recv(count)
            if not packet:
                return None
            data += packet
        return data


mySocket = None


def DoConnect():
    global mySocket
    if not mySocket:
        try:
            mySocket = SocketAdaptor()
            if not mySocket.isConnected:
                raise RuntimeError("Unable to connect to NeuroExplorer. Is NeuroExplorer running?")
        except Exception as ex:
            raise RuntimeError("Unable to connect to NeuroExplorer. Is NeuroExplorer running? Error: " + str(ex))


def nexRunCommand(theCommand):
    global mySocket
    DoConnect()

    mySocket.sendMessage(theCommand)
    msg = mySocket.receiveMessage()
    return msg


# simple command dict here so that we do not have to modify nex.py
def buildFunctionCommandDict(functionName, pars):
    commandDict = {"type": "ExecuteNexScriptCommand", "functionName": functionName, "parameters": []}
    for key, value in pars.items():
        if isinstance(value, bytes):
            value = value.encode('utf-8')
        commandDict["parameters"].append({key: value})
    return commandDict


# simple result processing here so that we do not have to modify nex.py
def processResultFromNex(resultJsonBytes):
    if not resultJsonBytes:
        raise RuntimeError("invalid return from NeuroExplorer")
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
            from nex import NexDoc
            return NexDoc(returnedObject)
        if returnedObject['type'] == 'varRef':
            from nex import NexVar
            return NexVar(returnedObject)

        # special cases with vectors, etc.

        if returnedObject['type'] == 'vectorsInFile':
            filePath = returnedObject["filePath"]
            n = returnedObject['numberOfValues']
            theFile = open(filePath, 'rb')
            if not theFile:
                raise RuntimeError("unable to open temp file")
            a = array.array('d')
            a.fromfile(theFile, n)
            theFile.close()
            os.remove(returnedObject["filePath"])
            columnSize = returnedObject['columnSize']
            if columnSize == 0:
                return a.tolist()
            numColumns = len(a) / columnSize
            matrix = []
            for col in range(int(numColumns)):
                column = array.array('d', a[col * columnSize:(col + 1) * columnSize])
                matrix.append(column.tolist())
            return matrix

        if returnedObject['type'] == 'emptyVectors':
            numVectors = returnedObject['numVectors']
            matrix = []
            for i in range(int(numVectors)):
                matrix.append([])
            return matrix

        if returnedObject['type'] == 'vectors':
            data = returnedObject['value']
            a = array.array('d')
            decoded = base64.decodebytes(data.encode('utf-8'))
            a.frombytes(decoded)
            columnSize = returnedObject['columnSize']
            if columnSize == 0:
                return a.tolist()
            numColumns = len(a) / columnSize
            matrix = []
            for col in range(int(numColumns)):
                column = array.array('d', a[col * columnSize:(col + 1) * columnSize])
                matrix.append(column.tolist())

            return matrix

        if returnedObject['type'] == 'vectorsOfStrings':
            return returnedObject['value']

    return returnedObject


def RunJsonCommand(pars, functionName):
    commandDict = buildFunctionCommandDict(functionName, pars)
    resultJsonBytes = nexRunCommand(json.dumps(commandDict))
    return processResultFromNex(resultJsonBytes)


def nexGetProperty(varOrDoc, theId, propertyName):
    return RunJsonCommand(locals(), "GetProperty")


def nexGetIndexedNumResValue(docId, row, col):
    commandDict = {"type": "ExecuteNexScriptCommand", "functionName": "GetNumRes", "parameters": []}
    fakeDoc = {'type': 'docRef', 'value': docId}
    commandDict["parameters"].append({'doc': fakeDoc})
    commandDict["parameters"].append({'row': row})
    commandDict["parameters"].append({'col': col})
    resultJsonBytes = nexRunCommand(json.dumps(commandDict))
    return processResultFromNex(resultJsonBytes)


def nexAddContValue(varId, timestamp, value):
    commandDict = {"type": "ExecuteNexScriptCommand", "functionName": "AddContValue", "parameters": []}
    fakeVar = {'type': 'varRef', 'value': varId}
    commandDict["parameters"].append({'var': fakeVar})
    commandDict["parameters"].append({'timestamp': timestamp})
    commandDict["parameters"].append({'value': value})
    resultJsonBytes = nexRunCommand(json.dumps(commandDict))
    return processResultFromNex(resultJsonBytes)


def nexAddInterval(varId, interval_start, interval_end):
    commandDict = {"type": "ExecuteNexScriptCommand", "functionName": "AddInterval", "parameters": []}
    fakeVar = {'type': 'varRef', 'value': varId}
    commandDict["parameters"].append({'var': fakeVar})
    commandDict["parameters"].append({'interval_start': interval_start})
    commandDict["parameters"].append({'interval_end': interval_end})
    resultJsonBytes = nexRunCommand(json.dumps(commandDict))
    return processResultFromNex(resultJsonBytes)


def nexAddTimestamp(varId, timestamp):
    commandDict = {"type": "ExecuteNexScriptCommand", "functionName": "AddTimestamp", "parameters": []}
    fakeVar = {'type': 'varRef', 'value': varId}
    commandDict["parameters"].append({'var': fakeVar})
    commandDict["parameters"].append({'timestamp': timestamp})
    resultJsonBytes = nexRunCommand(json.dumps(commandDict))
    return processResultFromNex(resultJsonBytes)


def nexCopyValuesToDocVar(docId, index, varId):
    return RunJsonCommand(locals(), "CopyValuesToDocVar")


def nexSetDocProperty(docId, propertyAsJson):
    return RunJsonCommand(locals(), "SetDocProperty")


def nexGetIndexedValue(varId, index):
    return RunJsonCommand(locals(), "GetIndexedValue")


def nexSetContValues(varId, timestamps, values):
    if len(timestamps) != len(values):
        raise ValueError('timestamps and values should be the lists of the same length')

    if len(timestamps) < 2000:
        return RunJsonCommand(locals(), "SetContVarValues")
    else:
        ts = array.array('d')
        ts.fromlist(timestamps)
        v = array.array('d')
        v.fromlist(values)

        fd, temp_path = tempfile.mkstemp()
        os.write(fd, ts.tobytes())
        os.write(fd, v.tobytes())
        os.close(fd)
        return RunJsonCommand({'varId':varId, 'tempFile':temp_path}, "SetContVarValues")


def nexSetIndexedValue(varId, index, theValue):
    return RunJsonCommand(locals(), "SetIndexedValue")


def nexSpectrum(values):
    return RunJsonCommand(locals(), "Spectrum")


def nexSetTimestamps(varId, timestamps):
    return RunJsonCommand(locals(), "SetTimestamps")


def nexCreateWaveformVar(docId, name, waveformSamplingRate, timestamps, values):
    return RunJsonCommand(locals(), "CreateWaveformVar")


def nexJsonStringAndNumArray(jsonString, numList):
    return RunJsonCommand(locals(), "JsonStringAndNumArray")

