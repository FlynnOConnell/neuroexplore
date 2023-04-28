from __future__ import print_function, division
import sys
sys.path.append(r'C:\ProgramData\Nex Technologies\NeuroExplorer 5 x64')
import nex

file = r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs\SFN13_2018-12-10_RS.nex'

doc = nex.OpenDocument(file)

doc["T3_Citric_Acid"] = doc["T2_Citric_Acid"]
doc["T3_NaCl"] = doc["T2_NaCl"]
doc["T3_MSG"] = doc["T2_MSG"]
doc["T3_Quinine"] = doc["T2_Quinine"]
doc["T3_Sucrose"] = doc["T2_Sucrose"]
doc["T3_ArtSal"] = doc["T2_ArtSal"]

nex.SaveDocument(doc)
nex.CloseDocument(doc)

print("Preprocessing Complete")
