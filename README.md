# Disaster Response Pipeline Project

### Dependencies
- Latest version of Python
- ML libraries : NumPy, SciPy, Pandas, Sciki-Learn
- NLP libraries : NLTK
- SQLite Database libraries : SQLalchemy
- Web App and Data Visualization : Flask, Ploty

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Additional files
1. ETL pipeline preperation notebook inside data folder will help in understanding the implemented ETL pipeline.
2. ML pipeline preperation notebook inside models folder will help in understanding the implemented ML pipeline.

### Acknowledgements
Figure Eight for providing messages dataset to traim the model.
