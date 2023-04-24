
from pathlib import Path
from nex.nexfile import Reader
from params import Params
from data_utils.file_handling import parse_filename

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')
reader = Reader(useNumpy=True)
params = Params()

neurons = []
files_check = []
for file_path in datadir.glob('*.nex'):

    session_info = parse_filename(file_path.name)
    fileData = reader.ReadNexFile(str(file_path))
    neuron_check = []
    unsorted = False

    for var in fileData['Variables']:
        if var['Header']['Type'] == 0:
            name = var['Header']['Name']
            if ("UNS" in name) or ("J" in name):
                files_check.append(file_path.name)
                unsorted = True
                continue
            else:
                neuron_check.append(name)

    if len(neuron_check) > 1:
        files_check.append(file_path.name)
    else:
        neurons.extend(neuron_check)

files_check = list(set(files_check))
