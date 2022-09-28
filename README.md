# covid_19_switzerland
This repository contains the data and the codes used to generate the results shown in the paper: **COVID-19 in Switzerland real-time epidemiological1 analyses powered by EpiGraphHub** 


The data saved as `.csv` in the `data_article` folder comes from the [Epigraphhub platform](epigraphhub.org). You can look at the code used to extract this data in the `download_data.ipynb` notebook.

The code that came from the `epigraphhub` Python package refers to the version `1.2.0` of the platform. 

The repo are organized as follow: 
* The `spatial_analysis.ipynb` notebook show the code to create the Figures 1, 4, 5 and 6 of the article. 
* The `correlation_and_inference.ipynb` notebook show the code to create the Figures 2, 3, 7 and 8 of the article. 
* The `forecast_models.ipynb` notebook show the code to create the Figures 9 and how the metrics in Table 2 was computed.  