# covid_19_switzerland
This repository contains the data and the code used to generate the results shown in the paper: **COVID-19 in Switzerland real-time epidemiological analyses powered by EpiGraphHub** Published on Nature Scientific Data.


The data saved as `CSV` files  in the `data_article` folder come from the [Epigraphhub platform](https://dash.epigraphhub.org). You can look at the code used to extract this data in the `download_data.ipynb` notebook.

The code that came from the `epigraphhub` Python package refers to the version `1.2.0` of the platform. 

The repo is  organized as follows:
 
* The `spatial_analysis.ipynb` notebook shows how to create figures 1, 4, 5 and 6 of the article. 
* The `correlation_and_inference.ipynb` notebook shows how to create  figures 2, 3, 7 and 8 of the article. 
* The `forecast_models.ipynb` notebook shows how to create figures 9 and how the metrics in Table 2 were computed.  