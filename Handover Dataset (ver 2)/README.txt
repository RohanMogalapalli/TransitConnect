README

Dataset Title:
A Pilot LTE Drive-Test Dataset for Handover and Mobility Analysis in Urban Bangladesh

Version 2.0

Repository Description
----------------------
This repository contains a publicly available LTE drive-test dataset collected in a live commercial LTE network in Dhaka, Bangladesh. The dataset includes raw diagnostic logs, Layer-3 Measurement Reports, Event Statistics, parent datasets extracted from the diagnostic logs, and processed datasets prepared following the preprocessing methodology.

The repository is intended to promote transparency, reproducibility, and independent analysis of LTE radio measurements and handover behavior.

----------------------------------------------------------------------
Repository Structure
----------------------------------------------------------------------

Handover Dataset/
│
├── DRM Files/
│   ├── DRM Files_Dataset_1/--- 7 drm files
│   ├── DRM Files_Dataset_2/--- 2 drm files
│   └── DRM Files_Dataset_3/--- 3 drm files
│
├── Event Statistics/
│   ├── Event Statistics_Dataset_1/--- 7 csv files
│   ├── Event Statistics_Dataset_2/--- 2 csv files
│   └── Event Statistics_Dataset_3/--- 3 csv files
│
├── Measurement Reports/
│   ├── Measurement Report_Dataset_1/--- 7 text files
│   ├── Measurement Report_Dataset_2/--- 2 text files
│   └── Measurement Report_Dataset_3/--- 3 text files
│
├── Parent Dataset/
│   ├── Dataset_1/--- 7 csv files
│   ├── Dataset_2/--- 2 csv files
│   └── Dataset_3/--- 3 csv files
│
└── Processed Dataset/
    ├── Dataset_1_ver_1.csv
    ├── Dataset_1_ver_2.csv
    ├── Dataset_2_ver_1.csv
    ├── Dataset_2_ver_2.csv
    ├── Dataset_3_ver_1.csv
    └── Dataset_4_ver_2.csv
----------------------------------------------------------------------
Folder Description
----------------------------------------------------------------------

1. DRM Files

This folder contains the original XCAL-M Diagnostic Record Message (DRM) files collected during the LTE drive tests.

Dataset composition:

• DRM Files_Dataset_1 contains 7 DRM files.
• DRM Files_Dataset_2 contains 2 DRM files.
• DRM Files_Dataset_3 contains 3 DRM files.

These files represent the original measurement data acquired directly during the drive tests and remain unmodified.

The DRM files can be opened using XCAL-M software to inspect the complete measurement logs and extract additional radio or protocol-layer information beyond the variables included in this dataset.

----------------------------------------------------------------------
2. Measurement Reports

Each DRM file has a corresponding Layer-3 Measurement Report exported from XCAL-M.

The Measurement Reports contain decoded LTE Radio Resource Control (RRC) signalling messages, including:

• Measurement Reports
• Paging messages
• RRC Reconfiguration messages
• Serving-cell measurements
• Neighbour-cell measurements

These files enable users to examine the signalling procedures associated with LTE mobility and handover events and provide transparency regarding the origin of the processed datasets.

----------------------------------------------------------------------
3. Event Statistics

Each DRM recording has a corresponding Event Statistics file.

These files summarize LTE mobility-related events extracted by XCAL-M, including:

• Event timestamp
• Event type
• Frequency information
• Base/interface type
• Event result
• Handover attempt
• Handover success
• Handover failure
• Event count

The Event Statistics files can be used to analyse handover behaviour, mobility events, and network performance independently of the processed datasets.

----------------------------------------------------------------------
4. Parent Dataset

The Parent Dataset contains the structured tabular data extracted directly from the Measurement Reports before preprocessing.

The parent datasets preserve the original extracted measurements and include serving-cell and neighbour-cell radio measurements together with mobility-related information collected during the drive tests.

These datasets represent the source from which the processed datasets were generated.

----------------------------------------------------------------------
5. Processed Dataset

The processed datasets were generated from the Parent Dataset following the preprocessing workflow.

The preprocessing includes:

• Timestamp alignment
• Measurement association
• Conversion of RSRP and RSRQ from physical units to standardized 3GPP Information Element (IE) values
• Handover event labelling
• Removal of prolonged intervals without neighbour-cell measurements
• Calculation of representative CINR values

The processed datasets are intended to provide a clean and structured dataset suitable for statistical analysis, reinforcement-learning-based handover studies, and small-scale machine learning experiments.

----------------------------------------------------------------------
Relationship Among Files
----------------------------------------------------------------------

The repository follows the processing pipeline below:

DRM Files
      │
      ▼
Measurement Reports
      │
      ▼
Parent Dataset
      │
      ▼
Processed Dataset

Event Statistics are generated independently from the DRM files and provide event-level summaries that complement the measurement datasets.

Each DRM recording has corresponding:

• Measurement Report
• Event Statistics file
• Parent Dataset file

allowing complete traceability from the processed datasets back to the original drive-test recordings.

----------------------------------------------------------------------
Software Requirements
----------------------------------------------------------------------

Raw DRM files:

• XCAL-M O3.5.2.64N

CSV datasets:

Any standard spreadsheet or data analysis software can be used, including:

• Microsoft Excel
• MATLAB
• Python
• R

No custom preprocessing software or scripts are required to use the released datasets.

----------------------------------------------------------------------
Recommended Workflow
----------------------------------------------------------------------

Recommended usage of the repository is as follows:

1. Inspect the original DRM files using XCAL-M (optional).

2. Review the Layer-3 Measurement Reports to understand LTE signalling procedures and radio measurements.

3. Examine the Event Statistics files to analyse mobility events and handover performance.

4. Use the Parent Dataset to perform custom preprocessing if required.

5. Use the Processed Dataset for statistical analysis, visualization, reinforcement-learning experiments, or small-scale machine learning studies.

Users wishing to implement alternative preprocessing strategies are encouraged to begin from the Parent Dataset or the original DRM files.

----------------------------------------------------------------------
Dataset Version
----------------------------------------------------------------------

This repository corresponds to Version 2.0 of the dataset.

Future corrections, documentation updates, or dataset extensions will be released as new repository versions through Mendeley Data to maintain version traceability and reproducibility.

----------------------------------------------------------------------
Citation
----------------------------------------------------------------------

If this dataset is used in academic work, please cite the associated Data in Brief publication together with the Mendeley Data repository DOI.

----------------------------------------------------------------------
Contact
----------------------------------------------------------------------

For questions regarding the dataset, please contact the corresponding author listed in the associated publication.