____________________________________________________
# A Boosted Model for Predicting Tremor, Dyskinesia, and Medication Levels:

# I.  Introduction:

Parkinson’s disease (PD) is a neurodegenerative disease that primarily affects the motor system but also exhibits other symptoms. Typical motor symptoms of the disease include tremors, slowness (bradykinesia), posture and walking perturbations, muscle rigidity and speech perturbations. Additionally, patients may experience side effects from their medication in the form of dyskinesia (involuntary movement). In the clinic, symptoms and side-effects are evaluated using physician observation and patient reports, however this infrequent assessment may not reflect a patient's typical state on an average day. Mobile sensor (accelerometers) may be useful for tracking a patient's symptom severity and response to medication, and have the potential to provide more detailed information that patients and physicians can use to make care decisions.

#### Project Origins:
[BEAT-PD DREAM Challenge]    (https://www.synapse.org/#!Synapse:syn20825169/wiki/)

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
- sklearn , lightgbm , catboost , matplotlib , seaborn , mpl_toolkits.mplot3d , 

# III. Project Architecture:

This repository [PD_Challenge](https://github.com/ashuabhi-github/Pd_Challenge) includes three main directories and 3 files:

### III-1. Directories:

├── activity_labels.txt <br />
├── features_info.txt <br />
├── features.txt <br />
├── README.txt <br />
├── Data Extraction.ipynb <br />
├── Part1-till hurst-CISPD.ipynb <br />
├── Part1-till hurst-RealPD.ipynb <br />
├── Part2-wavelet features-CISPD.ipynb <br />
├── Part2-wavelet features-REALPD.ipynb <br />
├── Part3-last set of features CISPD.ipynb <br />
├── Part3-last set of features RealPD.ipynb <br />
├── Modeling.ipynb <br />


**Before running any code please add these zipped data and csv files into the parent directory** <br />
├── cis-pd.ancillary_data.tar.bz2 <br />
├── cis-pd.clinical_data.tar.bz2 <br />
├── cis-pd.data_labels.tar.bz2 <br />
├── cis-pd.testing_data.tar.bz2 <br />
├── cis-pd.training_data.tar.bz2 <br />
├── real-pd.ancillary_data.tar.bz2 <br />
├── real-pd.clinical_data.tar.bz2 <br />
├── real-pd.data_labels.tar.bz2 <br />
├── real-pd.testing_data.tar.bz2 <br />
├── real-pd.training_data.tar.bz2 <br />
├── cis-pd.CIS-PD_Test_Data_IDs.csv <br />
├── real-pd.REAL-PD_Test_Data_IDs.csv <br />
├── BEAT-PD_SC1_OnOff_Submission_Template.csv <br />
├── BEAT-PD_SC2_Dyskinesia_Submission_Template.csv <br />
└── BEAT-PD_SC3_Tremor_Submission_Template.csv <br />


**Exported training datas from Part1,Part2,Part3 of CISPD & REALPD ipynb** <br />
├── analysis2_cispd_comp_training_abhiroop_tillhurst.csv <br />
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartphone.csv <br />
├── analysis2_realpd_comp_training_abhiroop_tillhurst_smartphone.csv <br />
├── cispd_wavelet_training_features.csv <br />
├── realpd_wavelet_features_smartphone_training.csv <br />
├── realpd_wavelet_features_smartwatch_training.csv <br />
├── cispd_comp_training_abhiroop_lastfeatures.csv <br />
├── realpd_comp_training_abhiroop_lastfeatures_smartphone.csv <br />
└── realpd_comp_training_abhiroop_lastfeatures_smartwatch.csv <br />

**Exported testing datas from Part1,Part2,Part3 of CISPD & REALPD ipynb** <br />
├── analysis2_cispd_comp_testing_abhiroop_tillhurst.csv <br />
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv <br />
├── analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv <br />
├── cispd_wavelet_testing_features.csv <br />
├── realpd_wavelet_features_smartphone_testing.csv <br />
├── realpd_wavelet_features_smartwatch_testing.csv <br />
├── cispd_comp_testing_abhiroop_lastfeatures.csv <br />
├── realpd_comp_testing_abhiroop_lastfeatures_smartphone.csv <br />
└── realpd_comp_testing_abhiroop_lastfeatures_smartwatch.csv <br />

**Exported clinical datas from Part3 of CISPD & REALPD ipynb** <br />
├── cispd_clinical_preprocessed.csv <br />
└── realpd_clinical_preprocessed.csv <br />

**Note:Modeling.ipynb will use these exported csv files to create machine learning models** <br />
		
### III-2. Notebooks:

     - `Data Extraction.ipynb`: This is the notebook file extract signals from these zip folders 
    

	- `Part1-till hurst-CISPD.ipynb`: This notebook file contains the signal processing and time and frequency domain features with pipeline for Cispd patient.
    - `Part1-till hurst-RealPD.ipynb`: This notebook file contains the signal processing time and frequency domain features pipeline for Realpd patient.
    
	- `Part2-wavelet features-CISPD.ipynb`:  This notebook file contains the wavelet features for Cispd patient.
    - `Part2-wavelet features-REALPD.ipynb`:  This notebook file contains the wavelet features for Realpd patient.
    
 	- `Part3-last set of features CISPD.ipynb`: This notebook file contains the signal processing DWT features and few other time and frequency domain feartures with pipeline for Cispd patient.
    - `Part3-last set of features RealPD.ipynb`: This notebook file contains the signal processing DWT features and few other time and frequency domain feartures with pipeline for Realpd patient.
  
     - `Modeling.ipynb`: This is the notebook file which  related to machine learning and detailed analysis of each step performed to build the final model
	
	- `README.md` : It contains a short description of this project and necessary steps to 
	                run it successfully.


# IV. Datasets and Inputs:
## IV-1. Original Datasets
### . The general process:

### A. The Experiment:
The experiments were carried out with a group of 30 volunteers within an age bracket of 19-48 years. They performed a protocol of activities composed of six basic activities: three static postures (`standing`, `sitting`, `lying`) and three dynamic activities (`walking`, `walking downstairs` and `walking upstairs`). The experiment also included postural transitions that occurred between the static postures. These are: `stand-to-sit`, `sit-to-stand`, `sit-to-lie`, `lie-to-sit`, `stand-to-lie`, and `lie-to-stand`. 
All the participants were wearing a smartphone (Samsung Galaxy S II) on the waist during the experiment execution. They captured `3-axial linear acceleration` and `3-axial angular velocity `at a constant rate of `50Hz` using the embedded accelerometer and gyroscope of the device. The experiments were video-recorded to label the data manually. The resulted dataset stored in the directory `Raw-Data` could be considered as the original labelled dataset and from it different subsets would be generated.


![image2]


|  User 01 inertial signals|
| :--------: |
| ![image4]  |


| Walking| Standing | 
| :--------:  |:------:	|
| ![image5]  	|  ![image6] |


| Sitting| Lying| 
| :--------:  |:------:	|
|  ![image7]  |![image8] |



### B. Data Splitting:
This `Raw-Data` was randomly partitioned into two sets, where `70%` of the volunteers was selected for generating the training data and `30%` for the test data.


![image10]


### C. Signal Processing:

**Noise filtering:**
The features selected for this database come from the accelerometer and gyroscope 3-axial raw signals `t_Acc-XYZ` and `t_Gyro-XYZ`. These time domain signals (**prefix 't' to denote time**) were captured at a constant rate of `50 Hz`. Then they were filtered using a median filter and a `3rd order low pass Butterworth filter` with a corner frequency of `20 Hz` to remove noise. 

**1- Median Filter:**  was applied to reduce background noise.


![image9]


**2- 3rd order Low pass Butterworth filter** with a cut-off, `frequency = 20hz` was applied to remove high frequency noise.

Resulted Signals are:  `total_acc_XYZ` and `Gyro_XYZ`.

##### Gravity filtering:
Similarly, the acceleration signal `total-acc-XYZ` was then separated into body and gravity acceleration signals `(tBodyAcc-XYZ and tGravityAcc-XYZ)` using another `low pass Butterworth filter` with a corner frequency of `0.3 Hz`. Since the gravitational force is assumed to have only low frequency components.


Resulted components are:  `total_acc-XYZ` ==> `tBody_acc-XYZ` + `tGravity_acc-XYZ`

### D. Activity Selection:
From the processed signals presented in the picture above, they extract data points and activity labels related only to the first six activities which are: `standing`, `sitting`, `lying`, `walking`, `walking downstairs` and `walking upstairs` called Basic Activites(BA) to build the first version of this dataset.
**The second version of this datasets includes all the `12 activities` : Basic Activities (BA) and Postural Transitions (PT).

### E. Feature Selection:
Subsequently, the body linear acceleration and angular velocity were derived in time to obtain Jerk signals (`tBodyAccJerk-XYZ` and `tBodyGyroJerk-XYZ`). Also, the magnitude of these three-dimensional signals was calculated using the Euclidean norm (`tBodyAccMag`, `tGravityAccMag`, `tBodyAccJerkMag`, `tBodyGyroMag`, `tBodyGyroJerkMag`). 
Finally, a Fast Fourier Transform `(FFT)` was applied to some of these signals producing `fBodyAcc-XYZ`, `fBodyAccJerk-XYZ`, `fBodyGyro-XYZ`, `fBodyAccJerkMag`, `fBodyGyroMag`, `fBodyGyroJerkMag`.
**The `f` to indicate frequency domain signals.** 

### F. Windowing:
These Signals were then sampled in fixed-width sliding windows of `2.56` sec and `50%` overlap `(128 readings/window)`.

### G. Features Generation:
From each sampled window, a vector of 561 features was obtained by calculating variables from the time and frequency domain.

-------------------------------------------------------------------------------------------------
### [Human Activity Recognition Using Smartphones Dataset Version 1.0](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#)
**Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Alessandro Ghio(1), Luca Oneto(1) and Xavier Parra(2):**

**[1] -** `Smartlab - Non-Linear Complex Systems Laboratory DITEN - Università  degli Studi di Genova, Genoa (I-16145), Italy.` 

**[2] -** `CETpD - Technical Research Centre for Dependency Care and Autonomous Living Universitat Politècnica de Catalunya (BarcelonaTech). Vilanova i la Geltrú (08800), Spain.`

**@mail:** `activityrecognition@smartlab.ws`

This first version is located in `./Data/Original-Data/UCI-HAR-Dataset/`. As mentioned earlier this dataset concerns only **Basic Activities** performed by the users during the experiments. It includes two main directories and they can be used separately.

**Dataset Architecture:** 

Under The `UCI-HAR-Dataset` Directory we have:
 
 1. ` ./Inertial Signals/`: This directory includes the **Semi-processed features** of this version.


		- ` ./Inertial-Signals/train/`:  The train folder includes 11 files.

			- `total_acc_x_train.txt`: The acceleration signal from the smartphone accelerometer X axis 
			                           in standard gravity unit 'g'. Every row shows a 128-element vector.
						   The same description applies for the `total_acc_y_train.txt` and                    
						   `total_acc_z_train.txt` files for the Y and Z axis. 

			- `body_acc_x_train.txt`: The body acceleration signal obtained by subtracting the gravity 
			                           from the total acceleration. The same description applies for the
						   `body_acc_y_train.txt` and `body_acc_z_train.txt` files for the Y 
						   and Z axis.

			- `body_gyro_acc_x_train.txt`: The angular velocity vector measured by the gyroscope for each
			                               window sample. The units are radians/second. The same description 
						       applies for the `body_gyro_y_train.txt` and `body_gyro_z_train.txt`
						       files for the Y and Z axis. 


		- ` ./Inertial-Signals/test/*`: This folder includes necessary testing files of inertial signals 
		                                following the same analogy as in `./Inertial Signals/train/`.



2. `./Processed-Data/` : This directory includes the **fully processed features** which concerns the same six activities. 


		- `X_train.txt`: Train features, each line is composed 561-feature vector with time and 
		                 frequency domain variables.

		- `X_test.txt`: Test features, each line is composed 561-feature vector with time and 
		                frequency domain variables.
				
		- `features_info.txt`: Shows information about the variables used on the feature vector.

		- `features.txt`: includes list of all 561 features


- `y_train.txt`: train activity labels, Its range is from 1 to 6

- `y_test.txt`: test activity labels, Its range is from 1 to 6

- `subject_train.txt`: training subject identifiers, Its range is from 1 to 30

- `subject_test.txt`: testing subject identifiers, Its range is from 1 to 30

- `activity_labels.txt`:

- `README.md`:

**Licence:**

Use of this dataset in publications must be acknowledged by referencing the following publication [1]:  

[1]- `Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.` 


**Note:** This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

----------------------------------------------------------------------------------------------------------
### [Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set Version 2.1]( https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

**Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Luca Oneto(1) and Xavier Parra(2)**

[1] - `Smartlab, DIBRIS - Università  degli Studi di Genova, Genoa (16145), Italy.` 

[2] - `CETpD - Universitat Politècnica de Catalunya. Vilanova i la Geltrú (08800), Spain.`

@mail: **`har@smartlab.ws`**

Website: **`www.smartlab.ws`**

This dataset is an extended version of the [UCI Human Activity Recognition Using smartphones Dataset V 1.0](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) mentioned earlier.
This version provides the original signals captured from the smartphone sensors mentioned earlier `Raw-Data`, instead of the ones semi-processed and sampled into windows located in `Inertial-signals` directory which were provided in version 1.0. This change was done in order to be able to make online tests with the data. Moreover, the activity labels were updated in order to include postural transitions that were not part of the previous version of the dataset. This Dataset includes two main directories and they can be used separately.

**Dataset Architecture:**

Under The `HAPT-Dataset` Directory we have: 

1. `./Raw-Data/`:  This the original data (unprocessed data) obtained directly from the set of experiments after activity labelling . It includes Raw triaxial signals from the accelerometer and gyroscope of all trials with participants. This Directory includes 123 files:
		
		- **61 acc files are**: `Raw-Data/acc_expXX_userYY.txt`: The raw triaxial acceleration signal
		                         for the experiment number XX and associated to the user number YY.
				         Every row is one acceleration sample (three axis)[X,Y,Z] captured 
				         at a frequency of 50Hz. 
		
		- **61 gyro files are**: `Raw-Data/gyro_expXX_userYY.txt`: The raw triaxial angular
		                          speed signal for the experiment number XX and associated to 
					  the user number YY. Every row is one angular velocity sample 
					  (three axis)[X,Y,Z] captured at a frequency of 50Hz. 
		
		- **One last file is** :`RawData/labels.txt`: includes labels of all the performed 
		                         activities (1 per row). 
			- Column 1: experiment number ID, 
			- Column 2: user number ID,
			- Column 3: activity number ID 
			- Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
			- Column 5: Label end point (in number of signal log samples)



| Raw Data Samples| 
| :--------:  |
|![image3] |

| Number of rows per Experience|Number of rows per activity  | 
| :--------:  |:------:	|
| ![image11]  	|  ![image12] |

| Number of useful rows per Experience| Mean time in seconds per Activity | 
| :--------:  |:------:	|
| ![image13]  	|  ![image14] |



2. `./ Processed-Data/`: This Directory includes the **fully processed dataset** which concerns all The twelve activities performed by users in the experiment.
		
		- `X_train.txt` : Train features, each line is composed 561-feature vector with time and 
		                  frequency domain variables.
		
		- `X_test.txt` : Test features, each line is composed 561-feature vector with time and 
		                 frequency domain variables.
		
		- `y_train.txt`: train activity labels, its range is from 1 to 12
		
		- `y_test.txt`: test activity labels, its range is from 1 to 12
		
		- `subject_id_train.txt`: training subject identifiers, Its range is from 1 to 30
		
		- `subject_id_test.txt`: testing subject identifiers, Its range is from 1 to 30
		
		- `features_info.txt`: Shows information about the variables used on the feature vector.
		
		- `features.txt`: includes list of all 561 features

- `activity_labels.txt`:
- `README.md`:

**License:**

Use of this dataset in publications must be acknowledged by referencing the following publications **[1]**:

**[1]** - `Jorge-L. Reyes-Ortiz, Luca Oneto, Albert Samà, Xavier Parra, Davide Anguita. Transition-Aware Human Activity Recognition Using Smartphones. Neurocomputing. Springer 2015.`

**Note:** This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.


## IV-2. New Datasets:

The New Datasets are the results of applying my own signal processing pipeline to Raw-Data. This directory includes one main folder ` full_Datasets_type_I_and_II `:

**`Type I `:** to denote V 1.0 which includes Basic activities only.

**`Type II `:** to denote V 2.1 which includes Both Basic Activities and Postural Transitions.

- ` ./New-Data/full_Datasets_type_I_and_II/ `: includes Datasets produced by the signal processing pipeline using ` Raw-Data ` only. All parts are fully processed. Each line(observation) includes XXX features + the **subject_Id** and the **activity label** related the observation.
		
		- ` Dataset_I_part1.csv `: The first 5000 rows of Dataset type I 
		
		- ` Dataset_I_part2.csv `: The rest of rows of Dataset type I 
		
		- ` Dataset_II_part1.csv `: The first 6000 rows of Dataset type II
		
		- ` Dataset_II_part2.csv `: The rest of rows of Dataset type II 

	- `new features info.txt `: Includes info about features(columns) in both Datasets type I and type II
	
	- `new_features.txt `: the full list of features (column names) of Dataset type I and II

# V. Running the code:

In the Terminal or Command Prompt, navigate to the folder containing the project files which contains this `README.md` and then use the command `jupyter notebook file_name.ipynb` to open up a browser window or tab to work with your notebook. Alternatively, you can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens.

If you want to generate RawData statistics and overwrite the new datasets type I and II you should run the `Part_I--Signal-Processing-Pipeline.ipynb`. 

If you want to generate new datasets type I and II statistics and its machine learning results you should run the second notebook `Part_II--Machine-Learning-Part.ipynb`.

**Notes:**
- Please before running any notebook pay attention to running durations first (durations were computed on a laptop core i5 8Gbs of RAM).

- The Data owners didn't provide the signal processing code. The signal processing pipeline was developed using only explanations provided in both original datasets (Version 1.O and 2.1). As a result The General Process mentioned earlier is a little bit different from the one coded in the signal processing pipeline.

- The windowing methods used in the signal processing pipeline differs depending on the type of the data generated. Reports included in this reporsitery provides detailed information about these differences.

Please If you have any ambiguity about any part of this project or just want to have more info about this subject don't hestitate to contact me at: `abdessamad.anass@gmail.com` 














# Pd_Challenge

## Step1: Unzipp these folders by running Data Extraction.ipynb :- ##
 * real-pd.training_data.tar.bz2
 * real-pd.testing_data.tar.bz2
 * real-pd.REAL-PD_Test_Data_IDs.csv
 * real-pd.data_labels.tar.bz2
 * real-pd.clinical_data.tar.bz2
 * real-pd.ancillary_data.tar.bz2
 * cis-pd.training_data.tar.bz2
 * cis-pd.testing_data.tar.bz2
 * cis-pd.data_labels.tar.bz2
 * cis-pd.clinical_data.tar.bz2
 * cis-pd.CIS-PD_Test_Data_IDs.csv
 * cis-pd.ancillary_data.tar.bz2

## Step2: Run Part1 Python Codes:- ##
 * Part1-till hurst-CISPD.ipynb
 * Exported Data:
     * analysis2_cispd_comp_training_abhiroop_tillhurst.csv
     * analysis2_cispd_comp_testing_abhiroop_tillhurst.csv
 * Part1-till hurst-RealPD.ipynb
 * Exported Data:
     * analysis2_realpd_comp_testing_abhiroop_tillhurst_smartphone.csv
     * analysis2_realpd_comp_testing_abhiroop_tillhurst_smartwatch.csv
     * analysis2_realpd_comp_training_abhiroop_tillhurst_smartphone.csv
     * analysis2_realpd_comp_training_abhiroop_tillhurst_smartwatch.csv
    
## Step3: Run Part2 Python Codes:- ##
 * Part2-wavelet features-CISPD.ipynb
 * Exported Data:
     * cispd_wavelet_training_features.csv
     * cispd_wavelet_testing_features.csv
 * Part2-wavelet features-REALPD.ipynb
 * Exported Data:
     * realpd_wavelet_features_smartphone_testing.csv
     * realpd_wavelet_features_smartphone_training.csv
     * realpd_wavelet_features_smartwatch_testing.csv
     * realpd_wavelet_features_smartwatch_training.csv
     * realpd_wavelet_testing_features.csv
  
## Step4: Run Part3 Python Codes:- ##
 * Part3-last set of features CISPD.ipynb
 * Exported Data:
     * cispd_comp_training_abhiroop_lastfeatures.csv
     * cispd_comp_testing_abhiroop_lastfeatures.csv
     * cispd_clinical_preprocessed.csv
 * Part3-last set of features RealPD.ipynb
 * Exported Data:
     * realpd_comp_training_abhiroop_lastfeatures_smartphone.csv
     * realpd_comp_testing_abhiroop_lastfeatures_smartphone.csv
     * realpd_comp_training_abhiroop_lastfeatures_smartwatch.csv
     * realpd_comp_testing_abhiroop_lastfeatures_smartwatch.csv
     * realpd_clinical_preprocessed.csv
  
## Step5: Run Modeling Codes:- ##
 * This python code(Modeling.ipynb) will use all exported csv files for prediction.
  
 
 Note: Output of Step2-4 is a feature vector, which will further be used in Step5 for modeling
