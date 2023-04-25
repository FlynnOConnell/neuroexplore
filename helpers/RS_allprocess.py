
from __future__ import print_function, division
import sys
sys.path.append('C:\\ProgramData\\Nex Technologies\\NeuroExplorer 5 x64')
import nex
import math


doc = nex.GetActiveDocument()

#Make InterLick Trials
doc["Lick_Int"] = nex.MakeIntervals(doc["Lick"],  - 0.001, 0.001)
doc["LickOpposite_Int"] = nex.IntOpposite(doc, doc["Lick_Int"])
doc["Interbout_Int"] = nex.IntSize(doc["LickOpposite_Int"], 1, 1100000)
doc["Lick_bout_Int"] = nex.IntOpposite(doc, doc["Interbout_Int"])
doc["LB_Start"] = nex.StartOfInterval(doc["Lick_bout_Int"])
doc["LB_End"] = nex.EndOfInterval(doc["Lick_bout_Int"])
doc["IB_Start"] = nex.StartOfInterval(doc["Interbout_Int"])
doc["IB_End"] = nex.EndOfInterval(doc["Interbout_Int"])
doc["Spontaneous"] = nex.IntSize(doc["LickOpposite_Int"], 10, 1100000)
doc["SpontRev"] = nex.IntOpposite(doc, doc["Spontaneous"])
nex.Delete(doc, doc["Lick_Int"])
nex.Delete(doc, doc["LickOpposite_Int"])
nex.Delete(doc, doc["Interbout_Int"])
nex.Delete(doc, doc["Lick_bout_Int"])

#Make Tastants Easy to Access
doc["T1_A1"] = nex.FirstAfter(doc["Citric_Acid"], doc["drylick"],  - 1, 600)
doc["T1_N1"] = nex.FirstAfter(doc["NaCl"], doc["drylick"],  - 1, 600)
doc["T1_M1"] = nex.FirstAfter(doc["MSG"], doc["drylick"],  - 1, 600)
doc["T1_Q1"] = nex.FirstAfter(doc["Quinine"], doc["drylick"],  - 1, 600)
doc["T1_S1"] = nex.FirstAfter(doc["Sucrose"], doc["drylick"],  - 1, 600)
doc["T1_AS1"] = nex.FirstAfter(doc["ArtSal"], doc["drylick"],  - 1, 600)
doc["T2_Citric_Acid"] = doc["Citric_Acid"]
doc["T2_NaCl"] = doc["NaCl"]
doc["T2_MSG"] = doc["MSG"]
doc["T2_Quinine"] = doc["Quinine"]
doc["T2_Sucrose"] = doc["Sucrose"]
doc["T2_ArtSal"] = doc["ArtSal"]

nex.SaveDocument(doc)

print("Preprocessing Complete")
