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
