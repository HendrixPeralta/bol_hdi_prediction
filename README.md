# Spatial Monitoring of Development Indicators in Bolivia: Municipal-Level
Evidence Leveraging Machine Learning and Remote Sensing Data.

This repository contains the code and resources for a project aimed at leveraging remote sensing data and machine learning techniques to predict Sustainable Development Goal (SDG) indicators in Bolivian municipalities. The project demonstrates the integration of spatial data analysis with machine learning methodologies to provide actionable insights for regional development and policy-making.

## Project Overview

### Objective
The primary goal of this project is to assess the predictive power of remote sensing data for Sustainable Development Goal (SDG) indicators at the municipal level. Coupled with the prediction of the SDGs, by employing machine learning techniques, the project aims to:
- Understand spatial disparities in SDG achievement.
- Cluster municipalities into regions based on their SDG achievement.
- Provide data-driven recommendations for policy interventions.

### Data
The project utilizes:
- **Remote sensing data**: Extracted from satellite imagery using Google Earth Engine (GEE) and platforms such as AidData and the GEE Community, focusing on environmental, infrastructure, socioeconomic, accessibility, geographical, and land cover variables.
- **SDG indicators**: Sourced from municipal-level datasets in Bolivia, covering 15 dimensions of development. 

### Methodology
1. **Spatial Analysis**:
   - Identification of spatial dependence existence in Bolivia. 
   - Creation of spatial development clusters for each of the development dimensions. 

2. **Clustering and Regionalization**:
   - Machine learning clustering techniques such as **K-Means** and **Hierarchical Clustering** for regional classification.
   - Identification of the region’s challenges regarding SDGs. 
	
3. **SDG Prediction**:
   - Prediction of SDG indexes using machine learning models such as **Ridge Regression**, **Lasso Regression**, **ElasticNet Regression**
   - Identification of relevant predictors. 

4. **Evaluation**:
   - Model performance evaluated using metrics like RMSE and R².
   - Validation through cross-validation techniques.

### Results
Key findings include strong spatial dependence across SDG indicators, significant regional disparities (with Region 2 being the least developed), and the effectiveness of remote sensing data in predicting SDGs like No Poverty (SDG 1) and Climate Action (SDG 13). 
However, limitations were noted for SDGs like Life on Land (SDG 15), emphasizing the need for complementary data sources. 

Finally, Socioeconomic factors, such as nighttime lights and population density, emerged as critical predictors. The study's methodological originality lies in its combination of spatial analysis, clustering, and predictive modeling, highlighting the importance of considering spillover effects in policy design.

## Key Features
- Integration of spatial econometrics and machine learning.
- Use of clustering techniques for regional classification.
- Comprehensive analysis pipeline from data preprocessing to model evaluation.

## Technologies Used
- **Programming Languages**: Python, JavaScript (for GEE).
- **Libraries**: pandas, NumPy, scikit-learn, seaborn, matplotlib, geopandas.
- **Tools**: Google Earth Engine, Geoquery.

## Future Work
- Expand the dataset to include additional remote sensing variables.
- Explore deep learning models for improved predictions.
- Conduct longitudinal analysis to assess temporal changes in SDG indicators.
- Collaborate with policymakers to validate findings and develop actionable recommendations.
