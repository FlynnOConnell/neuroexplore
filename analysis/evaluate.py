import pstats
import webbrowser
import tempfile
import os

# Load the pstat file profile
p = pstats.Stats(r'C:\Users\Flynn\AppData\Local\JetBrains\PyCharm2022.3\snapshots\neuroexplore.pstat')

# Create a temporary HTML file to hold the snakeviz output
fd, path = tempfile.mkstemp('.html')
os.close(fd)

# Generate the snakeviz output to the HTML file
p.sort_stats('cumulative').print_stats()
p.dump_stats(path)

# Open the HTML file in a web browser using the `webbrowser` module
webbrowser.open('file://' + path)