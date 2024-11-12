Predicting baseball pitches through classification using MLB Statcast data


Background: 

There's no advantage in baseball quite like knowing what pitch is coming. Whether the pitcher is unintentionally giving away their pitches (aka "tipping"), or whether the batting team is using some elaborate scheme to steal signs, there is always a desire for some kind of knowledge about the next pitch. One of the hardest parts of hitting is adjusting to the different speeds and spins on the fly. Most methods of stealing signs are illegal, or at the least frowned upon. With the sport rapidly incorporating more and more analytics, it may be beneficial for hitters to know certain patterns for pitchers they're facing by knowing the most likely pitch. 

Research Question:

How accurately, if at all, can the next pitch type be predicted for an MLB pitcher?

Data:

The raw data inputs are PitchFX pitch-level data from MLB Baseball Savant. This raw data is obtained through the pybaseball python package, since licenses to this data are not publically available. The pybaseball package can query MLB Baseball Savant through a handful of functions upon importing

Process:

Data Exploration
- Explore API functions and data retrieved
- Vizualize distributions and summary statistics

Data Cleaning
- Clean data types
- Remove/impute missing rows and columns
- Remove irrelevant columns

Data Transformation
- Outliers
- Encode categorical variables
- Load pull_data.py script with repeatable functions to pull, clean, and transform data

Feature Selection
- TBD
  
Model Exploration/CV
- KNN
- Random Forest
- TBD

Model Selection
- TBD

Model Training/Testing
- TBD

Model Application/Deployment
-TBD









