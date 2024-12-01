**Predicting baseball pitches through classification using MLB Statcast data**


Background: 

There's no advantage in baseball quite like knowing what pitch is coming. Whether the pitcher is unintentionally giving away their pitches (aka "tipping"), or whether the batting team is using some elaborate scheme to steal signs, there is always a desire for some kind of knowledge about the next pitch. One of the hardest parts of hitting is adjusting to the different speeds and spins on the fly. Most methods of stealing signs are illegal, or at the least frowned upon. With the sport rapidly incorporating more and more analytics, it may be beneficial for hitters to know certain patterns for pitchers they're facing by knowing the most likely pitch. 

Research Question:

How accurately, if at all, can the next pitch type be predicted for an MLB pitcher?

Data:

The raw data inputs are PitchFX pitch-level data from MLB Baseball Savant. This raw data is obtained through the pybaseball python package, since licenses to this data are not publically available. The pybaseball package can query MLB Baseball Savant through a handful of functions upon importing

Approach:

The first task will be to understand the pybaseball API and form a cohesive script to pull data quickly for the model. The API can be a bit slow since it’s pulling live data from MLB Baseball Savant. When considering what timeframe to consider when pulling pitch data, it seems 1 season is the most common sense choice. This makes the most sense for speed purposes, and also due to pitchers sometimes changing repertoires between seasons. 

After pulling in the raw data, a substantial amount of cleaning and transforming will be needed. In terms of features, Baseball Savant basically gives you every metric you could think of on a pitch-level basis. Most of it however, won’t be necessary for a predictive pitch model. Many of these features are describing the event itself (break angle, swing speed, location, launch angle etc.). I narrowed down to just any features that could possibly predict the next pitch. Further feature selection will take place during modeling. I then added in some lag features for previous pitches, since pitches are often thrown in sequences. 

The modeling approach will be to test a variety of multiclass classification models and select the most accurate while considering which model fits the problem best. Different models will be compared based on their accuracy, F1-score, and then visually with confusion matrices and ROC curves. A new model will need to be generated on a per-pitcher basis, since all pitchers have different tendencies and slightly different pitch types as well. For simplicity, certain pitch types will be grouped together and mapped accordingly. This will allow us to more easily compare results between pitchers. 
