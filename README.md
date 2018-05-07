# ECG Features
A library for extracting a wide range of features from single-lead ECG waveforms.

In the 2017 Physionet Challenge, competitors were asked to build a model to classify a
single lead ECG waveform as either Normal Sinus Rhythm, Atrial Fibrillation, Other
Rhythm, or Noisy. The database consisted of 12,186 ECG waveforms that were donated by 
AliveCor. Data were acquired by patients using one of three generations of AliveCor's 
single-channel ECG device. Waveforms were recorded for an average of 30 seconds with 
the shortest waveform being 9 seconds, and the longest waveform being 61 seconds. The 
figure below presents examples of each rhythm class and the AliveCor acquisition device.

![Rock Image](figures/waveform_examples.png)

This repository the code feature extraction code we used for our submission. 

## License
[MIT](LICENSE.txt)