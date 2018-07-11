# ECG Features
A library for extracting a wide range of features from single-lead ECG waveforms. These feature are grouped into three
main categories: (1) Template Features, (2) RR Interval Features, and (3) Full Waveform Features. This repository 
contains the feature extraction code we used for our submission to the 2017 Physionet Challenge. 

## Dataset
In the 2017 Physionet Challenge, competitors were asked to build a model to classify a single lead ECG waveform as 
either Normal Sinus Rhythm, Atrial Fibrillation, Other Rhythm, or Noisy. The dataset consisted of 12,186 ECG waveforms 
that were donated by AliveCor. Data were acquired by patients using one of three generations of AliveCor's 
single-channel ECG device. Waveforms were recorded for an average of 30 seconds with the shortest waveform being 9 
seconds, and the longest waveform being 61 seconds. The figure below presents examples of each rhythm class and the 
AliveCor acquisition device.

![Waveform Image](figures/waveform_examples.png) 
*Left: AliveCor hand held ECG acquisition device. Right: Examples of ECG recording for each rhythm class.*

## Publications
1.	Goodfellow, S. D., A. Goodwin, R. Greer, P. C. Laussen, M. Mazwi, and D. Eytan (2018), Atrial fibrillation 
classification using step-by-step machine learning, Biomed. Phys. Eng. Express, 4, 045005. DOI: 10.1088/2057-1976/aabef4 

2. Goodfellow, S. D., A. Goodwin, R. Greer, P. C. Laussen, M. Mazwi, and D. Eytan, Classification of atrial fibrillation 
using multidisciplinary features and gradient boosting, Computing in Cardiology, Sept 24–27, 2017, Rennes, France.

## Affiliations
1. The Hospital for Sick Children <br>
Department of Critical Care Medicine  <br>
Toronto, Ontario, Canada

2. Laussen Labs <br>
www.laussenlabs.ca  <br>
Toronto, Ontario, Canada

## License
[MIT](LICENSE.txt)