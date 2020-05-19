____________________________________________________
# A Boosted Model for Predicting Tremor, Dyskinesia, and Medication Levels:

# I.  Introduction:

Parkinson’s disease (PD) is a neurodegenerative disease that primarily affects the motor system but also exhibits other symptoms. Typical motor symptoms of the disease include tremors, slowness (bradykinesia), posture and walking perturbations, muscle rigidity and speech perturbations. Additionally, patients may experience side effects from their medication in the form of dyskinesia (involuntary movement). In the clinic, symptoms and side-effects are evaluated using physician observation and patient reports, however this infrequent assessment may not reflect a patient's typical state on an average day. Mobile sensor (accelerometers) may be useful for tracking a patient's symptom severity and response to medication, and have the potential to provide more detailed information that patients and physicians can use to make care decisions.

#### Project Origins:
[BEAT-PD DREAM Challenge](https://www.synapse.org/#!Synapse:syn20825169/wiki/)

#### Project Overview:
We are provided with raw sensor (accelerometer and gyroscope) time series data recorded during the course of daily living, and will be asked to predict individuals' medication state and symptom severity.

#### Data Contributors:
Data for this project is provided by the Michael J. Fox Foundation and were collected by researchers at Northwestern University, University of Rochester, University of Alabama, University of Cincinnati and Radboud University Medical Center.

Data for the project have being hosted by the Brain Commons. The BRAIN Commons is a scalable cloud based platform for computational discovery designed for the brain health community.

#### Challenge Organizers:
Evidation (Luca Foschini)
Michael J. Fox Foundation (David Alonso, Mark Frasier, Julia Keefe, Luba Smolensky)
Northwestern University (Arun Jayaraman, Nicholas Shawen)
Radboud University Medical Center (Luc Evers)
Sage Bionetworks (Alex Mariakakis, Larsson Omberg, Solly Sieberts, Phil Snyder)


# II. Software Requirements:
This project requires **Python 3.7.3** and the following Python libraries installed:
- Python 3.7.3
- numpy  , pandas, scipy , math , glob , statsmodels.robust , spectrum , joblib , multiprocessing, time , warnings , pywt , 
- sklearn , lightgbm , catboost , matplotlib , seaborn , mpl_toolkits.mplot3d

# III. Project Architecture:

### III-1. Directories:

```bash
├── requirements.txt 
├── features_info.docx 
├── features.xlsx 
├── README.txt 
├── Data Extraction.ipynb 
├── Part1-till hurst-CISPD.ipynb 
├── Part1-till hurst-RealPD.ipynb 
├── Part2-wavelet features-CISPD.ipynb 
├── Part2-wavelet features-REALPD.ipynb 
├── Part3-last set of features CISPD.ipynb 
├── Part3-last set of features RealPD.ipynb 
├── Modeling.ipynb 
```

**Before running any code please add these zipped data and csv files into the parent directory** <br />
```bash
├── cis-pd.ancillary_data.tar.bz2 
├── cis-pd.clinical_data.tar.bz2 
├── cis-pd.data_labels.tar.bz2 
├── cis-pd.testing_data.tar.bz2 
├── cis-pd.training_data.tar.bz2 
├── real-pd.ancillary_data.tar.bz2 
├── real-pd.clinical_data.tar.bz2 
├── real-pd.data_labels.tar.bz2 
├── real-pd.testing_data.tar.bz2 
├── real-pd.training_data.tar.bz2 
├── cis-pd.CIS-PD_Test_Data_IDs.csv 
├── real-pd.REAL-PD_Test_Data_IDs.csv 
├── BEAT-PD_SC1_OnOff_Submission_Template.csv 
├── BEAT-PD_SC2_Dyskinesia_Submission_Template.csv 
└── BEAT-PD_SC3_Tremor_Submission_Template.csv 
```

**After running Data Extraction.ipynb these Dicretories will be created** <br />
```bash
├── ancillary_data 
| ├── smartphone_accelerometer 
| ├── smartwatch_accelerometer 
| └── smartwatch_gyroscope 
├── training_data 
| ├── smartphone_accelerometer 
| ├── smartwatch_accelerometer 
| └── smartwatch_gyroscope 
├── testing_data 
| ├── smartphone_accelerometer 
| ├── smartwatch_accelerometer 
| └── smartwatch_gyroscope 
├── clinical_data 
| ├── CIS-PD_Demographics.csv 
| ├── CIS-PD_UPDRS_Part1_2_4.csv 
| ├── CIS-PD_UPDRS_Part3.csv 
| ├── REAL-PD_Demographics.csv 
| ├── REAL-PD_Smartphone_Metadata.csv 
| ├── REAL-PD_UPDRS_Part1_2_4.csv 
| └── REAL-PD_UPDRS_Part3.csv 
```

**Exported training datas from Part1,Part2,Part3 of CISPD & REALPD ipynb** <br />
```bash
├── analysis2_cispd_comp_training_abhiroop_tillhurst.csv 
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartphone.csv 
├── analysis2_realpd_comp_training_abhiroop_tillhurst_smartphone.csv 
├── cispd_wavelet_training_features.csv 
├── realpd_wavelet_features_smartphone_training.csv 
├── realpd_wavelet_features_smartwatch_training.csv 
├── cispd_comp_training_abhiroop_lastfeatures.csv 
├── realpd_comp_training_abhiroop_lastfeatures_smartphone.csv 
└── realpd_comp_training_abhiroop_lastfeatures_smartwatch.csv 
```

**Exported testing datas from Part1,Part2,Part3 of CISPD & REALPD ipynb** <br />
```bash
├── analysis2_cispd_comp_testing_abhiroop_tillhurst.csv 
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv 
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv 
├── cispd_wavelet_testing_features.csv 
├── realpd_wavelet_features_smartphone_testing.csv 
├── realpd_wavelet_features_smartwatch_testing.csv 
├── cispd_comp_testing_abhiroop_lastfeatures.csv
├── realpd_comp_testing_abhiroop_lastfeatures_smartphone.csv
└── realpd_comp_testing_abhiroop_lastfeatures_smartwatch.csv
```

**Exported clinical datas from Part3 of CISPD & REALPD ipynb** <br />
```bash
├── cispd_clinical_preprocessed.csv
└── realpd_clinical_preprocessed.csv
```

**Note:Modeling.ipynb will use these exported csv files to create machine learning models** <br />
 <br />
### III-2. Notebooks Description:

```bash
1. Data Extraction.ipynb: Unzipp files
2. Part1-till hurst-CISPD.ipynb: Signal processing and time and frequency domain features pipeline for Cispd patient.
3. Part1-till hurst-RealPD.ipynb: Signal processing time and frequency domain features pipeline for Realpd patient.
4. Part2-wavelet features-CISPD.ipynb: Wavelet(CWT) features for Cispd patient.
5. Part2-wavelet features-REALPD.ipynb: Wavelet(CWT) features for Realpd patient.
6. Part3-last set of features CISPD.ipynb: Signal processing DWT features and few other time and frequency domain feartures with pipeline for Cispd patient.
7. Part3-last set of features RealPD.ipynb: Signal processing DWT features and few other time and frequency domain feartures with pipeline for Realpd patient.
8. Modeling.ipynb: Machine learning and detailed analysis of each step performed to build the final model
9. requirements.txt : All necessary libraries to run these code
10. features_info.docx : Explanation of features
11. features.xlsx : All features created for CISPD

9. README.md: Short description of this project and necessary steps to run it successfully.
```

# IV. Original Datasets
### A. The general process:
In this project, we are creating a machine-learning algorithm to predict the occurrence of tremors, dyskinesia, and medication state of the patients which using clinical data and signals that have been captured from various devices (smartphone, smartwatch). We did not find literature in the research community around to combine clinical and IoT data. There was extensive work done to extract new features from the provided data points. First, the signals were segregated into gravity and body motion components. After this, 2nd order and 3rd order features from sensor data were derived. Finally, we have then utilized an optimal set of features and modeled with Catboost.

### B. Data Splitting:
This `Raw-Data` was stratify partitioned into two sets, where `70%` of the measurement_id was selected for generating the training data and `30%` for the test data.

### C. Signal Processing:

**Noise filtering:**
The features selected for this database come from the accelerometer and gyroscope 3-axial raw signals `t_Acc-XYZ` and `t_Gyro-XYZ`. These time domain signals were captured at a constant rate of `50 Hz` for cispd and for realpd smartphone signals were captured ta rate of `100 Hz` and `50 Hz` for smartwatch. Then they were filtered using a `3rd order low pass Butterworth filter` with a corner frequency of `20 Hz` to remove noise.

Resulted Signals are:  `total_acc_XYZ` and `Gyro_XYZ`.

##### Gravity filtering:
Similarly, the acceleration signal `total-acc-XYZ` was then separated into body and gravity acceleration signals `(tBodyAcc-XYZ and tGravityAcc-XYZ)` using another `low pass Butterworth filter` with a corner frequency of `0.3 Hz`. Since the gravitational force is assumed to have only low frequency components.

Resulted components are:  `total_acc-XYZ` ==> `tBody_acc-XYZ` + `tGravity_acc-XYZ`

### D. Derived Signals:
Subsequently, the body linear acceleration and angular velocity were derived in time to obtain Jerk signals (`tBodyAccJerk-XYZ` and `tBodyGyroJerk-XYZ`). Also, the magnitude of these three-dimensional signals was calculated using the Euclidean norm (`tBodyAccMag`, `tGravityAccMag`, `tBodyAccJerkMag`, `tBodyGyroMag`, `tBodyGyroJerkMag`). 
Finally, a Fast Fourier Transform `(FFT)` was applied to some of these signals producing `fBodyAcc-XYZ`, `fBodyAccJerk-XYZ`, `fBodyGyro-XYZ`, `fBodyAccJerkMag`, `fBodyGyroMag`, `fBodyGyroJerkMag`.
**The `f` to indicate frequency domain signals.** 

### E. Features Generation:
From these derived signal, a vector of 908 features was obtained by calculating variables from the time and frequency domain.

-------------------------------------------------------------------------------------------------
# V. Running the code:

**Step1:** In the jupyter notebook or Command Prompt, navigate to the folder containing the project files which contains this `README.md`. Alternatively, you can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens. Put all those 15 zip files in the parent directory.

**Step2:** Run Data Extraction.ipynb, which will create these directories
- ancillary_data 
- training_data 
- testing_data 
- clinical_data 

**Step3:** Then Run Part1-till hurst-CISPD.ipynb and Part1-till hurst-RealPD.ipynb and then these files will be exported
- analysis2_cispd_comp_testing_abhiroop_tillhurst.csv 
- analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv 
- analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv 

**Step4:** Then Run Part2-wavelet features-CISPD.ipynb and Part2-wavelet features-REALPD.ipynb and then these files will be exported
- cispd_wavelet_testing_features.csv 
- realpd_wavelet_features_smartphone_testing.csv 
- realpd_wavelet_features_smartwatch_testing.csv 

**Step5:** Then Run Part3-last set of features CISPD.ipynb and Part3-last set of features RealPD.ipynb and then these files will be exported
- cispd_comp_testing_abhiroop_lastfeatures.csv
- realpd_comp_testing_abhiroop_lastfeatures_smartphone.csv
- realpd_comp_testing_abhiroop_lastfeatures_smartwatch.csv
- cispd_clinical_preprocessed.csv
- realpd_clinical_preprocessed.csv

**Step6:** Then Run Modeling.ipynb which will use these exported files for modeling and export these predicion files for three sub challenges
- final_tremor.csv (predicion for tremor)
- final_dyskinesia.csv (predicion for dyskinesia)
- final_on_off.csv (predicion for on_off)

**Notes:**
Please If you have any ambiguity about any part of this project or just want to have more info about this subject don't hestitate to contact me at: `abhiroopkumar.iitkgp@gmail.com / abhiroop.kumar@optum.com` 
