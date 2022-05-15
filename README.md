# Analysis of primary school interaction dataset by Barrat et al. [^1]

The current github repository contains all methods, code and data used to analyse the primary school interaction dataset by Barrat et al. [^1] for Sara Johanne Asche's master thesis on modelling the spread of COVID-19
in Primary schools.

## Contents

All the data analysed and output from the model can be found in the data folder. Furthermore figures generated are placed in fig_master. Since linear regression was plotted for all single nodes, the generated images are placed in the separate folder R2_img.

dataPreperation.py contains the documentation on how a 20-second contact dataset was condensed and accumulated into a network where the times two individuals interacted was added as a weight between the individuals. It also describes how new IDs were generated.

Functions in create_images.py are used to generate interaction and degree distributions, heatmaps, plot the graph, generate random networks, plot assortativity between day one and two and when investigating whether or not there was a bias for students to interact with other students of the same grade. The image below shows for instance the assortativity described in the thesis generated using `plotDegreeDegreeConnection()`.

![This is an image](https://github.com/SaraAsche/PrimarySchool/blob/master/fig_master/Assortativity2.png)

main_analysis_run.py contains most of the main analysis that has been run on the empiric primary school data.
Each nx.Graph objects is generated each time the file is run, based on an accumulated interaction csv generated in dataPreperation.py

temporal.py was used to look into lunch and interaction distribution for the different hours in the schoolday. Finally, the code in primaryAbstract.py is not used in the thesis, however it describes a simple abstract primary school model that fits well with COVID-19 Task force model's cliques.

[^1]:
    Alain Barrat, Ciro Cattuto, Alberto. E. Tozzi, Philippe Vanhems, and
    Nicholar Voirin. Measuring contact patterns with wearable sensors: methods, data characteristics and applications to data-driven simulations of infectious diseases. Clinical Microbiology and Infection, 20(1):10–16, 2014. doi: 10.1111/1469-0691.12472
