import nex  # Used to generate neuroexplorer commands
import os  # Used to go through directories
import json  # Used to read json files (data is stored in json files)

# Insert username below:
username = 'MyUsername'

_dir = "MyDirectory"  # Insert directory here

# Go to directory holding the JSON files
os.chdir(_dir % username)
# Get the names of all files in this directory
filelist = os.listdir('./')

# Convert all json files to nex files
for name in filelist:  # For all files in directory
    if name.endswith('.json'):  # If the file is a json file
        doc = nex.NewDocument(40000.)  # Use active document
        with open('R:\\Autobots Roll Out\\%s\\JSON_Files\\%s' % (username, name),
                  "r") as read_file:
            data = json.load(read_file)

        fileName = data['FileName'][:]  # Get file name
        print("Converting %s to a neuroexplorer format." % fileName)

        # Create event variables
        allEvents = data['Events']  # Get data for events
        for n in range(len(allEvents)):
            evtData = data['Events'][n]  # Get data for one event

            tempVar0 = str.split(str(evtData),
                                 '[')  # Split event data whereever "[" exists
            evtName = tempVar0[0][3:-3]  # Keep event name
            doc["Event"] = nex.NewEvent(doc, 0)  # Create new event with no timestamps

            tempVar1 = tempVar0[1][:-2]  # Keep only the timestamps for event
            tempVar2 = str.split(tempVar1,
                                 ', ')  # Split event timestamps whereever ", " exists
            for n in range(len(tempVar2)):  # For all timestamps
                evtTime = (float(tempVar2[n]))  # Convert from string to float
                nex.AddTimestamp(doc["Event"], evtTime)  # Add timestamps to event

            nex.Rename(doc, doc["Event"],
                       evtName)  # Change name of event to current event name
        print(name)

        # Create spike timestamps for each neuron
        allSpikes = data['Neurons']  # Get data for neurons
        for n in range(len(allSpikes)):
            spkData = data['Neurons'][n]  # Get data for one neuron

            tempVar0 = str.split(str(spkData),
                                 '[')  # Split spike data whereever "[" exists
            spkName = tempVar0[0][3:-3]  # Keep event name
            doc["Spike"] = nex.NewEvent(doc, 0)  # Create new spike with no timestamps

            tempVar1 = tempVar0[1][:-2]  # Keep only the timestamps for spike
            tempVar2 = str.split(tempVar1,
                                 ', ')  # Split spike timestamps whereever ", " exists
            spkTimes = []  # To hold float type timestamps
            b = [0] * len(tempVar2)
            for q in range(len(tempVar2)):  # For all timestamps
                spkTimes.append((float(tempVar2[
                                           q]) + 20) / 40000)  # Convert from string to float. Then add 20 (for time before spike onset) and divide by 40,000 kHz frequency
            for q in range(len(spkTimes)):  # For all timestamps
                b[q] = [spkTimes[q], q]  # Index timestamps
            b.sort()  # Sort indexed timestamps
            spkTimes.sort()  # Sort non-indexed timestamps as well
            for q in range(len(b)):  # For index
                b[q] = [b[q][1], q]  # keep only index and index the index
            b.sort()  # Resort, thereby obtaining the index of timestamp indexing
            for q in range(len(spkTimes)):  # For all timestamps
                nex.AddTimestamp(doc["Spike"], spkTimes[q])  # Add timestamps to spike

            nex.SetNeuronType(doc, doc["Spike"], 1)  # Convert spike to a neuron
            nex.Rename(doc, doc["Spike"],
                       spkName)  # Change name of neuron to current spike name

            # Create waveforms for each neuron
            # allWaves = data['Waveforms'] # Get all waveforms for all neuron
            # for n in range(len(allWaves)):
            waveData = data['Waveforms'][n]  # Get all waveforms for one neuron
            tempVar0 = str.split(str(waveData),
                                 '[')  # Split waveform data whereever "[" exists
            waveName = tempVar0[0][3:-3]  # Keep waveform name
            print(waveName)
            waves = [0] * (len(tempVar0) - 2)  # To hold all waveforms for a single neuron
            for w in range(2, len(tempVar0)):  # For all waveforms
                tempVar1 = tempVar0[w][:-3]  # Keep points for a single waveform
                tempVar2 = str.split(tempVar1,
                                     ', ')  # Split waveform points whereever ", " exists
                wavePoints = []  # To hold float type points
                for q in range(len(tempVar2)):  # For all waveform points
                    wavePoints.append(float(tempVar2[
                                                q]) / 1000)  # Convert from string to float. Then divide by 1,000 to convert from V to mV
                waves[w - 2] = [b[w - 2][1],
                                wavePoints]  # Append the timestamp indexing and all waveforms for neuron
            waves.sort()
            for q in range(len(waves)):
                waves[q] = waves[q][1]

            doc.CreateWaveformVariable(waveName, 40000, spkTimes,
                                       waves)  # Create waveform variable

        nexName = "%s.nex" % fileName  # Create neueroexoporer file name
        print(nexName)
        nex.SaveDocumentAs(doc, nexName)  # Save Nex file
        ndoc = nex.GetActiveDocument()  # Select Nex file
        nex.CloseDocument(ndoc)  # Close Nex file

print("Finished.")
