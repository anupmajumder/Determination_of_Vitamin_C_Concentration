# **Determination of Vitamin C Concentration Using NIR Spectroscopy and Chemometric Modeling**

## **Project Overview**
This project focuses on determining Vitamin C concentration using Near-Infrared (NIR) spectroscopy and chemometric modeling techniques. The goal is to analyze NIR spectral data and quantify Vitamin C concentration by applying chemometric methods. The script processes the spectral data to calculate and visualize the mean spectrum, and highlights key wavenumbers selected based on feature selection techniques.

## **Features**
- **Develop the chemometric pipeline**: Develop an optimized chemometric modeling framework for rapid, non-destructive quantification of Vitamin C (ascorbic acid) using Near-Infrared (NIR) spectroscopy
- **Prediction of vitamin C concentration**: This pipeline processes NIR spectral data, applies appropriate preprocessing techniques, selects the most informative wavelength intervals using the iPLS algorithm, and builds an optimized PLS regression model for accurate, non-destructive prediction of vitamin C concentration.
- **Mean Spectrum Calculation**: Computes the mean spectrum from the spectral data.
- **Wavenumber Highlighting**: Visualizes and marks the selected wavenumbers on the mean spectrum plot.
- **Plot Generation**: Generates visualizations of the mean spectrum and selected wavenumbers for analysis.

## **Dependencies**

To run this project, you will need the following Python libraries:

- `numpy`: For numerical operations and handling spectral data.
- `matplotlib`: For creating plots and visualizations.
- `pandas`: For handling and manipulating data (if required).
- `scipy`: For any advanced scientific calculations (if required).

You can install the necessary dependencies by running the following command:

```bash
pip install numpy matplotlib pandas scipy

## **How to Run the Project**

1. **Clone the repository** (or copy the script to your working directory):

   ```bash
   git clone <repository-url>
   ```

2. **Prepare your data**:
   - Ensure that you have your NIR spectral data in the form of a 2D NumPy array (`X`), where each row represents a sample and each column represents a wavenumber.
   - The `wvn` array should contain the corresponding wavenumbers for the spectral data.

3. **Running the script**:
   - After preparing your data, run the script that processes the NIR spectra and generates the plot.

4. **Visualize the results**:
   - Once the script is run, a plot will appear showing the mean spectrum and the selected wavenumbers highlighted in blue.

## **Example Output**

The script generates a plot of the mean spectrum with selected wavenumbers marked along the x-axis. The wavenumbers are highlighted using vertical dashed lines to indicate the most relevant features for Vitamin C concentration determination.
