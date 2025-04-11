# Predictive Policing Using Historical Crime and Weather Data

This project uses historical crime data from the City of Chicago along with weather data to forecast potential future crime hotspots in the city. The goal is to assist law enforcement or policymakers by predicting the probability of crime occurrences based on multiple temporal and environmental features.

## Data Sources

- **Crime Data:** [Crimes - 2001 to Present - City of Chicago](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)
  - Contains detailed records of reported crimes in Chicago from 2001 to the present.
  - Used fields: Date, Arrest, Latitude, Longitude, Community Area

- **Weather Data:** Collected using the `meteostat` Python library.
  - Based on hourly historical weather at O'Hare Airport (41.9742° N, 87.9073° W)
  - Fields used: temperature (`temp`), relative humidity (`rhum`), wind speed (`wspd`), precipitation (`prcp`), snowfall (`snow`)

## Process Overview

1. **Load & Clean Crime Data**
   - Loaded in chunks for memory efficiency
   - Parsed dates and rounded coordinates into 0.01 degree grids (~100 meters)

2. **Build Prediction Grid**
   - Created all combinations of time × location (hourly resolution)
   - Binned into a manageable grid to evaluate for possible future crimes

3. **Integrate Weather**
   - Fetched hourly weather for each `DateHour` in the dataset
   - Missing values filled using forward and backward propagation

4. **Feature Engineering**
   - Extracted features: hour, day of week, month, weekend flag
   - Added arrest-related rolling aggregates: RecentArrests, RepeatOffenderSignal

5. **Modeling**
   - Trained a RandomForestClassifier on historical features to classify if a crime occurred
   - Evaluated using standard classification metrics (ROC AUC, precision, recall)

6. **Forecasting**
   - Predicted next 12 hours after last timestamp in dataset
   - Created synthetic grid with expected weather and top repeat-offender hotspots
   - Output top 10 locations with highest predicted probability

7. **Visualization**
   - Displayed predictions using `folium` for map visualization
   - Markers indicate likely future crime events in real-world coordinates

## Limitations

- Weather data is from a single centralized station (not per grid cell)
- Predictions assume similar conditions will persist in the near future
- True real-time forecasting requires live crime feeds and micro-climate data

## Outputs

- PowerPoint Summary: `Predictive_Policing_Presentation.pptx`
- Jupyter Notebook: `predictive policing 1.1.ipynb`
- Cleaned Data Reference: `df_reference.pkl`



## ▶️ How to Run This Project

### 1. Set Up Environment
Make sure you have Python 3.8+ and install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib folium meteostat tqdm
```

You may use a virtual environment or conda:

```bash
conda create -n crime_env python=3.9
conda activate crime_env
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Start Jupyter Lab or Notebook and open the file:

```bash
jupyter lab
# or
jupyter notebook
```

Then open `predictive policing 1.1.ipynb` and run the cells sequentially.

### 3. Optional: Export Results

- Export predictions or plots directly from notebook.
- Use `joblib` to save/load models quickly.
- Final map will display inside notebook using Folium.

### Notes
- Ensure internet access is available to fetch weather data.
- Crime data CSV must be named exactly: `Crimes_-_2001_to_Present_20250410.csv`
- Output grid and forecast map appear toward the end of the notebook.

## Conda Environment Setup (Alternative to pip)

You can also use the provided `environment.yml` file to create a conda environment:

```bash
conda env create -f environment.yml
conda activate predictive_policing
```

This ensures all required packages and versions are correctly installed.

