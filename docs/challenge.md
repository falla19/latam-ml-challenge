# All parts Software Engineer (ML & LLMs) Challenge

## Overview

The first part of the **Software Engineer (ML & LLMs)** Application Challenge. In this, you will have the transcribed model from the .ipynb notebook to the model.py file. Each function in the model.py file is documented and explains how it works. In order to run the tests a few lines were added in the file and documented.

The **Logistic Regression model was chosen** to train and predict the data. Exploring the data, the following was found:
- The data is linear and the features are not correlated. 
    - Using a statistical test, such as the linearity test by scipy, the p-value of each attribute of the top 10 was less than 0.05, then the data to train is linear.
    - Using the .corr() function of a DataFrame, almost all the values were near 0, which led to no-correlated data.
- Logistic regression works better with linear data and no-correlated features, is also relatively fast to train and deploy.

Additionally, the following bugs were fixed:
- **exploration.ipynb**: All the code for the plots needed the 'x' and 'y' attributes to be declared in order to run. So, the attributes were included in each plot.
- **model.py**: The returning object of the preprocess function had a typo, the Union object was defined with parenthesis '()' instead of brackets '[]'. 

The second part of the **Software Engineer (ML & LLMs)** Application Challenge. In this, you will have the API to consume in order to predict the delay of some flights. The API receives an Object containing the flights to predict. If any of the columns values are incorrect it will raise an HttpException, otherwise return the delay prediction. In order to run the tests a few lines were added in the model.py file and documented.

Additionally, the following method were added to model.py file:
- **check_response**: Given the request predict the delay of the flights or return an error to raise an HttpException.

The third part of the **Software Engineer (ML & LLMs)** Application Challenge. In this, you will have the dockerized app deployed on GCP. The Dockerfile is using python 3.9 to mitigate errors and additional building wheels on packages like numpy and pandas. This version allow us to save time building the image and deploying it to GCP Container Registry. The following URL is the app deployed: https://challenge-service-prod-333009412295.us-central1.run.app

Running the stress test there were no fails and an average time of 700ms to resolve each request with 2487 requests at the time. The following shows the response time percentiles:

Response time percentiles (approximated)
 Type     Name                                                              50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 POST     /predict                                                          820    950    980    990   1000   1000   1100   1700   1800   1800   1800   2487
--------|------------------------------------------------------------|---------|------|------|------|------|------|------|------|------|------|------|------|
 None     Aggregated                                                        820    950    980    990   1000   1000   1100   1700   1800   1800   1800   2487

The fourth part of the **Software Engineer (ML & LLMs)** Application Challenge. In this, you will have CI/CD pipeline created and working with the branches develop and main. The idea behind this pipeline states the following:

### CI pipeline
- First of all, the pipeline will run when there's a 'push' in the develop branch. 
- The pipeline build a test Dockerfile.tests image to run the model test and api test.
- Then, it will build the Dockerfile image to push into the Artifact Registry provided by GCP. 
- Once is pushed, it will deploy the image using Cloud Run in a development service for testing.
- Then, it will build a test Dockerfile.stress image to run the stress test using the URL provided by the development service.
- If everything worked well, it will create a new branch release/**sha** and checkout to it.
- In the release branch, will create a PR to main branch.

### CD pipeline
- First of all, the pipeline will run when the PR made by the release branch is merged. With this, we ensure best practices on running the CD pipeline. 
- It will merge the PR made by the release branches. 
- The pipeline will build the Dockerfile image to push into the Artifact Registry provided by GCP. 
- Once is pushed, it will deploy the image using Cloud Run in a production service.
- If everything worked well, it will print the URL provided by the production service.

## Problem

A jupyter notebook (`exploration.ipynb`) has been provided with the work of a Data Scientist (from now on, the DS). The DS, trained a model to predict the probability of **delay** for a flight taking off or landing at SCL airport. The model was trained with public and real data, below we provide you with the description of the dataset:

|Column|Description|
|-----|-----------|
|`Fecha-I`|Scheduled date and time of the flight.|
|`Vlo-I`|Scheduled flight number.|
|`Ori-I`|Programmed origin city code.|
|`Des-I`|Programmed destination city code.|
|`Emp-I`|Scheduled flight airline code.|
|`Fecha-O`|Date and time of flight operation.|
|`Vlo-O`|Flight operation number of the flight.|
|`Ori-O`|Operation origin city code.|
|`Des-O`|Operation destination city code.|
|`Emp-O`|Airline code of the operated flight.|
|`DIA`|Day of the month of flight operation.|
|`MES`|Number of the month of operation of the flight.|
|`AÑO`|Year of flight operation.|
|`DIANOM`|Day of the week of flight operation.|
|`TIPOVUELO`|Type of flight, I =International, N =National.|
|`OPERA`|Name of the airline that operates.|
|`SIGLAORI`|Name city of origin.|
|`SIGLADES`|Destination city name.|

In addition, the DS considered relevant the creation of the following columns:

|Column|Description|
|-----|-----------|
|`high_season`|1 if `Date-I` is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.|
|`min_diff`|difference in minutes between `Date-O` and `Date-I`|
|`period_day`|morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59), based on `Date-I`.|
|`delay`|1 if `min_diff` > 15, 0 if not.|

## Challenge

### Context:

We need to operationalize the data science work for the airport team. For this, we have decided to enable an `API` in which they can consult the delay prediction of a flight.

*We recommend reading the entire challenge (all its parts) before you start developing.*

### Part I

In order to operationalize the model, transcribe the `.ipynb` file into the `model.py` file:

- If you find any bug, fix it.
- The DS proposed a few models in the end. Choose the best model at your discretion, argue why. **It is not necessary to make improvements to the model.**
- Apply all the good programming practices that you consider necessary in this item.
- The model should pass the tests by running `make model-test`.

> **Note:**
> - **You cannot** remove or change the name or arguments of **provided** methods.
> - **You can** change/complete the implementation of the provided methods.
> - **You can** create the extra classes and methods you deem necessary.

### Part II

Deploy the model in an `API` with `FastAPI` using the `api.py` file.

- The `API` should pass the tests by running `make api-test`.

> **Note:** 
> - **You cannot** use other framework.

### Part III

Deploy the `API` in your favorite cloud provider (we recomend to use GCP).

- Put the `API`'s url in the `Makefile` (`line 26`).
- The `API` should pass the tests by running `make stress-test`.

> **Note:** 
> - **It is important that the API is deployed until we review the tests.**

### Part IV

We are looking for a proper `CI/CD` implementation for this development.

- Create a new folder called `.github` and copy the `workflows` folder that we provided inside it.
- Complete both `ci.yml` and `cd.yml`(consider what you did in the previous parts).