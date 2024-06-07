# PlantTraits2024 ([Kaggle](https://www.kaggle.com/competitions/planttraits2024))

**Predict 6 plant traits from plant images and ancillary geodata in an effort to measure ecoysystem health**

## Contents

 - [Submissions](#Submissions)
 - [Project Overview](#Project-Overview)
 - [Data]()
   - [images](#Images)
   - [geodata](#Geodata)
   - [plant trait targets](#Plant-trait-targets)
 - [Preprocessing](#Preprocessing)
 - [Modeling](#Modeling)
 - [Outcomes](#Outcomes)
 - [Environment Notes](#Kaggle-Notebook-Environment)

### Submissions

 1. [Project Proposal](./proposal.pdf) *24.4.1*
 2. [Data Wrangling, EDA](./26.2.1%20Data%20Wrangling%20and%20EDA.ipynb) *26.2.1*
 3. [Preprocessing, Modeling](./28.1.1%20Preprocessing%20and%20Modeling.ipynb) *28.1.1*
 4. Documentation 
    - [report](./report.pdf) *28.1.2*
	- [slides](./slides.pdf) *28.1.3*
	- [model metrics](./model_metrics.csv)
	


### Project Overview

**[Kaggle Competition](https://www.kaggle.com/competitions/planttraits2024/overview)** | 
**[FGVC Workshop](https://sites.google.com/view/fgvc11/)** | 
**[Related Work](https://www.nature.com/articles/s41598-021-95616-0)**

*competition notes*
> The primary objective of this competition is to employ deep learning-based regression models, such as Convolutional Neural Networks (CNNs) like ConvNext or Transformers, to predict plant traits from photographs. These plant traits, although available for each image, may not yield exceptionally high accuracies due to the inherent heterogeneity of citizen science data. The various plant traits describe chemical tissue properties that are loosely related to the visible appearance of plants in images. Despite the anticipated moderate accuracies, the overarching goal is to explore the potential of this approach and gain insights into global changes affecting ecosystems. Your contribution to uncovering the wealth of data and the distribution of plant traits worldwide is invaluable.

*project notes*
 - use training dataset only
 - establish suitable evaluation metric for multi-output regression
 - final model should consider complexity and training time in addition to accuracy
 
### Data

#### Images

#### Geodata


#### Plant trait targets


### Preprocessing

- data cleaning, outlier detection
- feature normalization
- target transformation

### Modeling

### Outcomes


### Kaggle Notebook Environment

*my local setup requires DirectML for GPU + Tensorflow (v 1.15), so most GPU accelerated training occured within Kaggle notebooks. see environment description in file below.*

[Kaggle `pip freeze`](./kaggle_notebook_environment.txt)


