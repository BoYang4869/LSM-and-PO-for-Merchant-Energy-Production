////**************************************************************************************
//// Date: 03/31/2023
//// Author: Bo Yang, Selvaprabu Nadarajah, Nicola Secomandi
//// Description: C++ Implementation of Least Squares Monte Carlo for Merchant Energy Production
////              24-stage Ethanol Production: January Instance
////              High Order Polynomial Basis Functions
////**************************************************************************************

////***************************************************************************************************************************************
////              This cpp file uses the LAPACKE 3.7.1 package
////			  Please make sure that the package is properly installed before compiling and running this cpp file 
////              All functions are contained in this cpp file
////**************************************************************************************

//*********************************************
// Standard Headers
//*********************************************
#include <string>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <algorithm>

//*******************************
//External Headers
//*******************************
#include <lapacke.h>
#include <cblas.h>

//******************************************************************************
// Definitions of Initial Forward Structure and Loading coefficient Vectors
//******************************************************************************
using namespace std;
using namespace std::chrono;

//using namespace alglib;

#ifndef MULTCOMMTERMSTRUCTUREMODEL_HPP_
#define MULTCOMMTERMSTRUCTUREMODEL_HPP_

template<typename EXST>
struct struct_PriceModelInfo {
	//Input
	int nStages;
	int nCommodities;
	int nMarkets;
	int nMonths;
	int nDaysPerMonth;
	int nStartingMonth;
	int nSimSamples;
	int nRegSamples;
	int nRealFactors;
	int RndVarSeed;
	double timeDelta;
	double discFactor;
	EXST initForwCurves;

	int loadingCoeffMonths;
	int loadingCoeffMats;
	int loadingCoeffComm;
	int loadingCoeffMkts;
	vector<vector<EXST> > loadingCoeffs;

	//Data structures for antithetic variates
	int RndControl;
	vector<double> StoredRndVariable;
	vector<double> RndVariable;

	//Data structure containing forward curve
	vector<vector<vector<vector<EXST>>>> futForwCurve;
};
static int controlvalue = 0; // Control Variable for Antithetic Variates Method in Monte Carlo Simulation

//************************************************
// Parameters for the Ethanol Production Instance
//************************************************
string Month = "Jan"; // Jan instance. 
const int stage = 24; // 24 monthly stages
const int max_state = 3; // Each stage has at most 3 states "Operational", "Mothballed", and "Abandoned"
static const double production = 8.33; // Fixed Production Quantity
static const double corn_input_rate = 0.36; // Conversion rate of the corn
static const double NG_input_rate = 0.035; // Conversion rate of the natural gas
static const double cost_for_Abandonment = 0; // Fixed cost for the action "Abandon"
static const double cost_for_mothball = -0.5000; // Fixed cost for the action "Mothball"
static const double cost_for_reactivation = -2.5; // Fixed cost for the action "Reactivate"
static const double cost_for_production = -2.25; // Fixed cost for the action "Produce"
static const double cost_for_keepMothballing = -0.02917; // Fixed cost for the action "Keep Mothballing"
static const double cost_for_suspension = -0.5208; // Fixed cost for the action "Suspend"
double salvage_value = 0.0; // Salvage value for the abandonment
static const int time_lag_mothball = 1; // Time lag for the action "Mothball"
static const int time_lag_reactivation = 1; // Time lag for the action "Reactivate"

//*****************************************
// Parameters for the BCD Algorithm
//*****************************************
const int NumOfSamplePath = 70000; 
// Sample Number for Pathwise Linear Program

const int numofsamples = 100000; 
// Sample Number Unbiased Bound Simulation

const int nofcoeffs = 3 * (stage - 2) + 3 * 3 * (stage - 1);
// High Polynomial Basis Function Number at the Initial Stage

//***************************************************
// Definitions of Functions for Sample Generation
//***************************************************
double cpp_genalea(int *x0);
// Input a seed and return a random variable between 0 and 1

double randn(const double mu, const double sigma, int& seed);
// Generate a random variable following the standard normal distribution N(0,1)

void initializePriceModelStructures(ifstream &,
	ifstream &,
	vector<vector<vector<vector<double> > > > &,
	struct_PriceModelInfo<vector<vector<vector<double> > > > &); 
// Assign memory for the initial forward curve vector and loading coefficient vectors

void simulateForwardCurveAntitheticFM(
	struct_PriceModelInfo<vector<vector<vector<double> > > > &,
	vector<vector<vector<vector<double> > > >&);
// Generate a 24 stage forward curve

//***************************************************
// Definitions of Functions for LSM and Bound Simulation
//***************************************************
double rewardFunction(int, int, int, double, double, double);
//double rewardFunction(int state, int action, double conversion_margin);
// Return payoff for each feasible action

void basisFunction(int, int, vector<vector<vector<vector<double> > > >&);
// Compute basis function values for a forward curve sample

string itos(int i) { stringstream s; s << i; return s.str(); }
// Transfer Int to String

double PenaltyFunction(int, int, int, int, vector<vector<vector<vector<double>>>> &, vector<vector<vector<double>>> &, struct_PriceModelInfo<vector<vector<vector<double> > > > &);
// Return the dual penalty for each action. 
// Note that this function returns the penalty value for fixed Beta variable values

double greedypolicy(int, int, double discount_factor, vector<vector<vector<vector<double>>>> &, struct_PriceModelInfo<vector<vector<vector<double> > > > &);
// Return the greedy policy value for a sample path and a sequence of actions on that sample path

void lowerbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI);
// Return a unbiased lower bound estimate for a forward curve sample

double upperbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI);
// Return a unbiased dual (upper) bound estimate for a forward curve sample

double Continuation_Value(int action, int state, int current_stage, vector<vector<vector<vector<double>>>> &futures_price, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI);
// Return the value function approximation (VFA) for a particular stage and state
// with given Beta variables and a forward curve sample

void generation_A_matrix(int);
// Store a coefficient matrix for the LSM regression

//************************************************************************
// Definitions of Basis Function Vectors and Loading Coefficient Vectors
//************************************************************************
vector<vector<vector<double>>> coefficient;
// Store the solution of Beta variables in the PLP

vector<vector<vector<vector<vector<double>>>>> forwardcurve_Total;
// Store all forward curve samples

static double basis1[stage][3][stage] = { 0 };  
// Basis functions F_{i,j} for NG, corn, and ethanol 

static double basis2[stage][3][stage] = { 0 }; //
// Basis functions F_{i,j}^{2} for NG, corn, and ethanol

static double basis3[stage][stage][3] = { 0 }; //
// Basis functions F_{i,j}^{Corn} * F_{i,i+1}^{NG}, F_{i,j}^{Corn} * F_{i,i+1}^{E}, and F_{i,j}^{E} * F_{i,i+1}^{NG}

static double basis4[stage - 1][3][stage - 1] = { 0 }; //
// Basis functions F_{i,j} * F_{i,j+1} for NG, corn, and ethanol

double loadingcoeffs1[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j}^{2} 

double loadingcoeffs2[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j}^{Corn} * F_{i,i+1}^{NG}, F_{i,j}^{Corn} * F_{i,i+1}^{E}, and F_{i,j}^{E}

double loadingcoeffs3[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j} * F_{i,j+1} for NG, corn, and ethanol

//**************************
// Definitions of Stats 
//**************************
double AVERAGE = 0;
double diff_Upper_Bound = 0;
double diff_Lower_Bound = 0;
double variance_UpperBound = 0;
double variance_LowerBound = 0;

//***********************************************
// Definitions of Vectors for Bound Estimation
//***********************************************
double sample_Upper_Bound_Value[numofsamples] = { 0 };
// Store dual bound estimates on samples

double sample_Lower_Bound_Value[numofsamples] = { 0 };
// Store lower bound estimates on samples

float matrix_A[stage][NumOfSamplePath][nofcoeffs + 1] = { 0 };
// Store coefficient matrix for the LSM regression

double matrix_a[stage][(nofcoeffs + 1)*NumOfSamplePath] = { 0 };
// A copy of matrix_A in column major format

double matrix_b[stage][max_state][NumOfSamplePath][1] = { 0 };
// Store the RHS vector for the LSM regression

int zeroposition[nofcoeffs + 1] = { 0 };
// This vector stores the positions of columns with colinearity

static int action[stage] = { 0 };
// Store the action sequence of the greedy policy on a sample path

static int stateSet[stage] = { 0 };
// Store the state sequence of the greedy policy on a sample path
static double greedy_policy_value[stage] = { 0 };
// Store the state value of the greedy policy on a sample path

static double unbiased_lowerbound = 0;
// Lower bound estimate

static double unbiased_upperbound = 0;
// Dual bound estimate 
vector<vector<int>> statesForEachStage;
// Store the endogenous states for each stage

int main(int argc, char *argv[])
{
	try
	{
		auto start_time = high_resolution_clock::now();
		// "O" -- Operational; "A" -- Abandoned; "M1" -- Mothballed
		vector<int> vector_action_1 = { 'O' };
		vector<int> vector_action_2 = { 'O', 'A', 'M1' };
		vector<int> vector_action_3 = { 'O', 'A' };

		for (int i = 0; i < stage; i++)
		{
			if (i == 0)
			{
				statesForEachStage.push_back(vector_action_1);
			}
			else if (i >= stage - time_lag_reactivation)
			{
				statesForEachStage.push_back(vector_action_3);
			}
			else
			{
				statesForEachStage.push_back(vector_action_2);
			}
		}

		// Initialize the sampling process 
		vector<vector<vector<vector<double> > > > forwardCurveSample;
		struct_PriceModelInfo <vector<vector<vector<double>>>> tmpPMI;
	
		// Read data from files
		string InitialForwardInputPath = "./Ethanol_InputFile/priceModelInitCondnInputFile_" + Month + ".txt";
		ifstream priceModelCalibInputFile_0("./Ethanol_InputFile/priceModelCalibInputFile.txt", ios::in);
		ifstream priceModelInitCondnInputFile_0(InitialForwardInputPath, ios::in);

		initializePriceModelStructures(priceModelCalibInputFile_0,
			priceModelInitCondnInputFile_0,
			forwardCurveSample,
			tmpPMI);

		const double discount_factor = tmpPMI.discFactor;

		forwardcurve_Total.resize(NumOfSamplePath);
		for (int totalloop = 0; totalloop < NumOfSamplePath; totalloop++)
		{
			forwardcurve_Total[totalloop].resize(tmpPMI.nStages + 1);
			for (int t = 0; t < tmpPMI.nStages + 1; t++) {
				forwardcurve_Total[totalloop][t].resize(tmpPMI.nCommodities);
				for (int c = 0; c < tmpPMI.nCommodities; c++) {
					forwardcurve_Total[totalloop][t][c].resize(tmpPMI.nMarkets);
					for (int m = 0; m < tmpPMI.nMarkets; m++) {
						forwardcurve_Total[totalloop][t][c][m].resize(tmpPMI.nStages + 1);
						for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++){
							forwardcurve_Total[totalloop][t][c][m][maturity] = 0;
						}
					}
				}
			}
		}

		// Initialize the VFA vector
		coefficient.resize(stage);
		for (int i = 0; i < stage; i++)
		{
			coefficient[i].resize(max_state);
			for (int j = 0; j < max_state; j++)
			{
				coefficient[i][j].resize(nofcoeffs + 1);
				for (int k = 0; k < nofcoeffs + 1; k++)
				{
					coefficient[i][j][k] = 0;
				}
			}
		}

		// Generate a training sample set with 70,000 forward samples
		for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
		{
			cout << TotalLoop << endl;

			// start from Jan. Generate sample paths. loop.

			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample);

			for (int t = 0; t < tmpPMI.nStages + 1; t++) {
				for (int c = 0; c < tmpPMI.nCommodities; c++) {
					for (int m = 0; m < tmpPMI.nMarkets; m++) {
						for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++)
						{
							forwardcurve_Total[TotalLoop][t][c][m][maturity] = forwardCurveSample[t][c][m][maturity]; // forwardcurve stage + commodity + market + maturity
						}
					}
				}
			}

			// Generate basis functions based on the sample path
			basisFunction(0, stage, forwardcurve_Total[TotalLoop]);

			// Store the coefficient matrix for the LSM regression
			generation_A_matrix(TotalLoop);
		}

		// Check the multi-colinearity
		for (int sstage = 0; sstage < stage; sstage++){
			for (int k = 0; k < nofcoeffs; k++)
			{
				for (int kk = k + 1; kk < nofcoeffs + 1; kk++)
				{
					if (matrix_A[sstage][0][k] == matrix_A[sstage][0][kk])
					{
						for (int loop = 0; loop < NumOfSamplePath; loop++)
						{
							matrix_A[sstage][loop][kk] = 0;
						}
					}
				}
			}
		}

		// Perform the LSM regression
		auto LSM_start_time = high_resolution_clock::now();
		for (int i = stage - 1; i > 0; i--){
			for (int j = 0; j < statesForEachStage[i].size(); j++){
				int M = NumOfSamplePath;
				int N = 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1;
				int LDA = M;
				int LDB = M;
				int RHSN = 1;
				char TRANS = 'N';
				int count;
				int NumofColinearity;
				count = 0;
				for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++)
				{
					if (matrix_A[i][0][k] != 0)
					{
						for (int loop = 0; loop < NumOfSamplePath; loop++)
						{
							matrix_a[i][count] = matrix_A[i][loop][k];
							count++;
						}
					}
					else
					{
						zeroposition[k] = 1;
						N = N - 1;
					}
				}

				switch (statesForEachStage[i][j])
				{
				case 'A':
					break;
				case 'O':
					for (int loop = 0; loop < NumOfSamplePath; loop++)
					{
						basisFunction(0, stage, forwardcurve_Total[loop]);
						matrix_b[i][1][loop][0] = greedypolicy(i, 'O', tmpPMI.discFactor, forwardcurve_Total[loop], tmpPMI);
					}
					
					cout << "The stage is " << i << endl;

					LAPACKE_dgels(LAPACK_COL_MAJOR, TRANS, M, N, RHSN, matrix_a[i], M, *matrix_b[i][1], M);
		
					NumofColinearity = 0;
					
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++)
					{
						if (zeroposition[k] != 0)
						{
							cout << "Zeroposition is " << zeroposition[k] << endl;
							NumofColinearity++;
							coefficient[i][1][k] = 0;
						}
						else
						{
							coefficient[i][1][k] = matrix_b[i][1][k - NumofColinearity][0];
						}
					}

					for (int k = 0; k < nofcoeffs + 1; k++)
					{
						zeroposition[k] = 0;
					}

					break;
				case 'M1':
					for (int loop = 0; loop < NumOfSamplePath; loop++)
					{
						basisFunction(0, stage, forwardcurve_Total[loop]);
						matrix_b[i][2][loop][0] = greedypolicy(i, 'M1', tmpPMI.discFactor, forwardcurve_Total[loop], tmpPMI);
					}

					LAPACKE_dgels(LAPACK_COL_MAJOR, TRANS, M, N, RHSN, matrix_a[i], M, *matrix_b[i][2], M);

					NumofColinearity = 0;
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++)
					{
						if (zeroposition[k] != 0)
						{
							NumofColinearity++;
							coefficient[i][2][k] = 0;
						}
						else
						{
							coefficient[i][2][k] = matrix_b[i][2][k - NumofColinearity][0];
						}
					}
					for (int k = 0; k < nofcoeffs + 1; k++)
					{
						zeroposition[k] = 0;
					}
					break;
				}//endl of switch
			} // end of j
		} // end of i
		
		auto LSM_stop_time = high_resolution_clock::now();

		// Output the LSM VFA
		string SolutionPath = "./LSM_Beta_Coefficient_" + Month + ".txt";
		ofstream coeff_lsm(SolutionPath, ios::out);
		coeff_lsm.precision(20);
		for (int i = 1; i < stage; i++){
			if (i < stage - time_lag_reactivation){
				for (int j = 1; j < max_state; j++){
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++){
						coeff_lsm << coefficient[i][j][k] << endl;
					}
				}
			}
			else{
				for (int j = 1; j < max_state - 1; j++) { // three states needs approximation. 
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1) + 1; k++) {// PO model, do not constain constant i nthe basis functions 
						coeff_lsm << coefficient[i][j][k] << endl;
					}
				}
			}
		}
		coeff_lsm.close();

		//*******************************************************************
		// Simulate the unbiased lower and dual (upper) bounds
		//*******************************************************************
		vector<vector<vector<vector<double> > > > forwardCurveSample_1; // forward curve

		forwardCurveSample_1.resize(tmpPMI.nStages + 1);
		for (int t = 0; t < tmpPMI.nStages + 1; t++) {
			forwardCurveSample_1[t].resize(tmpPMI.nCommodities);
			for (int c = 0; c < tmpPMI.nCommodities; c++) {
				forwardCurveSample_1[t][c].resize(tmpPMI.nMarkets);
				for (int m = 0; m < tmpPMI.nMarkets; m++) {
					forwardCurveSample_1[t][c][m].resize(tmpPMI.nStages + 1);
					for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++)
					{
						// half of the forward curve is 0.  
						forwardCurveSample_1[t][c][m][maturity] = 0; // forwardcurve stage + commodity + market + maturity
					}
				}
			}
		}

		//**************************************************
		// Estimate the unbiased upper bound
		// Generate another sample set for the simulation, 
		// which is independent of the one for the training
		//**************************************************
		auto Upper_Bound_start_time = high_resolution_clock::now();
		tmpPMI.RndVarSeed = 6346538;

		for (int TotalLoop = 0; TotalLoop < numofsamples; TotalLoop++)
		{
			double OBJ = 0;

			cout << TotalLoop << endl;

			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample_1);

			int current_stage = 0;

			// Generate basis functions on a sample path
			basisFunction(current_stage, stage, forwardCurveSample_1);

			// Compute the dual bound estimate on a sample path
			OBJ = upperbound(tmpPMI.discFactor, forwardCurveSample_1, coefficient, tmpPMI);

			unbiased_upperbound += OBJ;

			sample_Upper_Bound_Value[TotalLoop] = OBJ;
		}// END OF TotalLOOP*

		auto Upper_Bound_End_time = high_resolution_clock::now();
		auto UB_duration = duration_cast<seconds>(Upper_Bound_End_time - Upper_Bound_start_time);
		cout << "Time for upper bound: " << UB_duration.count() << endl;
		cout << "unbiased upper bound: " << unbiased_upperbound / numofsamples << endl;

		//***********************************************
		// Generate the lower bound
		//************************************************
		auto Lower_Bound_Start_Time = high_resolution_clock::now();
		tmpPMI.RndVarSeed = 6346538;
		
		for (int TotalLoop = 0; TotalLoop < numofsamples; TotalLoop++)
		{
			cout << TotalLoop << endl;

			double sum = 0;

			// Generate a sample path
			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample_1);

			int current_stage = 0;

			// Generate basis functions on a sample path
			basisFunction(current_stage, stage, forwardCurveSample_1);

			stateSet[0] = 'O';

			// Compute the state and action sequences of the greedy policy
			lowerbound(tmpPMI.discFactor, forwardCurveSample_1, coefficient, tmpPMI);

			for (int i = 0; i < stage; i++)
			{
				greedy_policy_value[i] = rewardFunction(stateSet[i], action[i], i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0])*powf(discount_factor, i);
			}

			for (int Z = 0; Z < stage; Z++)
			{
				sum += greedy_policy_value[Z];
			}

			sample_Lower_Bound_Value[TotalLoop] = sum;
			unbiased_lowerbound += sum;	
		}// End of LOOP

		auto Lower_Bound_End_Time = high_resolution_clock::now();

		auto Total_End_Time = high_resolution_clock::now();

		unbiased_lowerbound = unbiased_lowerbound / numofsamples;
		
		unbiased_upperbound = unbiased_upperbound / numofsamples;
		
		//**********************************
		//Standard error for the dual bound
		//**********************************
		for (int i = 0; i < numofsamples; i++)
		{
			diff_Upper_Bound += (sample_Upper_Bound_Value[i] - unbiased_upperbound)*(sample_Upper_Bound_Value[i] - unbiased_upperbound);
		}
		variance_UpperBound = diff_Upper_Bound / (numofsamples - 1);
		variance_UpperBound = sqrt(variance_UpperBound);
		variance_UpperBound = variance_UpperBound / sqrt(numofsamples);

		//***********************************
		//Standard error for the lower bound
		//***********************************
		for (int i = 0; i < numofsamples; i++)
		{
			diff_Lower_Bound += (sample_Lower_Bound_Value[i] - unbiased_lowerbound)*(sample_Lower_Bound_Value[i] - unbiased_lowerbound);
		}
		variance_LowerBound = diff_Lower_Bound / (numofsamples - 1);
		variance_LowerBound = sqrt(variance_LowerBound);
		variance_LowerBound = variance_LowerBound / sqrt(numofsamples);

		//*******************************************************************************************
		// Print
		//*******************************************************************************************
		cout << "The estimated unbiased upper bound is " << unbiased_upperbound << endl;
		cout << "The SEM for unbiased Upper Bound is " << variance_UpperBound << endl;
		cout << "The estimated lower bound is " << unbiased_lowerbound << endl;
		cout << "The SEM for Lower Bound is " << variance_LowerBound << endl;

		auto Total_duration = duration_cast<seconds>(Total_End_Time - start_time);
		auto LSM_duration = duration_cast<seconds>(LSM_stop_time - LSM_start_time);
		auto LB_duration = duration_cast<seconds>(Lower_Bound_End_Time - Lower_Bound_Start_Time);

		cout << "The running time is " << Total_duration.count() << endl;
		cout << "Regression Time: " << LSM_duration.count() << endl;
		cout << "Lower Bound Time: " << LB_duration.count() << endl;
		cout << "Upper Bound Time: " << UB_duration.count() << endl;

		string FinalResultPaht = "./Result_LSM_" + Month + ".txt";
		ofstream finalResult(FinalResultPaht, ios::out);
		finalResult << "UB: " << unbiased_upperbound << endl;
		finalResult << "UB_SEM: " << variance_UpperBound << endl;
		finalResult << "LB: " << unbiased_lowerbound << endl;
		finalResult << "LB_SEM: " << variance_LowerBound << endl;
		finalResult << "Total Time: " << Total_duration.count() << endl;
		finalResult << "Regression Time: " << LSM_duration.count() << endl;
		finalResult << "Lower Bound Time: " << LB_duration.count() << endl;
		finalResult << "Upper Bound Time: " << UB_duration.count() << endl;
		finalResult.close();
	}
	catch (...)
	{
		cout << "Exception during optimization" << endl;
		system("pause");
	}
}

double rewardFunction(int state, int action, int current_stage, double spotPrice_Ethanol, double spotPrice_Corn, double spotPrice_NG)
{
	try
	{
		if (state == 'O'&& action == 'P'){
			return (spotPrice_Ethanol - (corn_input_rate)*spotPrice_Corn - (NG_input_rate)*spotPrice_NG)*production + cost_for_production;
		}//The order is important!
		if (state == 'O'&& action == 'A' && current_stage == stage - 1) {
			return salvage_value;
		}
		if (state == 'O'&& action == 'A') {
			//return cost_for_Abandonment + salvage_value;
			return cost_for_Abandonment;
		}
		if (state == 'O'&& action == 'M') {
			return cost_for_mothball;
		}
		if (state == 'O'&& action == 'S') {
			return cost_for_suspension;
		}
		if (state == 'M1' && action == 'M') {
			return cost_for_keepMothballing;
		}
		// current state is M2 (assume mothballing needs two periods). Actions can be A, M, R.
		if (state == 'M1'&& action == 'A') {
			//return salvage_value + cost_for_Abandonment;
			return cost_for_Abandonment;
		}
		if (state == 'M1' && action == 'R') {
			return cost_for_reactivation;
		}
		if (state == 'Empt' && action == 'Empt') {
			return 0;
		}
		if (state == 'NULL' && action == 'NULL') {
			return 0;
		}
	}
	catch (...)
	{
		cout << "error" << endl;
		cout << state << " " << action << " " << current_stage << endl;
	}
}

void basisFunction(int current_stage, int stage, vector<vector<vector<vector<double>>>> &futures_price) //generate all basis function for each stage.
{
	for (int v = current_stage; v < stage; v++) // v from 0 to 1 to 2 to 3
	{
		// F_i_j
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < stage - v; j++) // j from 0 to 1 to 2 to 3// basis function doesn't include spot price.
			{
				basis1[v][i][j] = futures_price[v][i][0][j];
			}
		}

		//(F_i_j)^2
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < stage - v; j++) // j from 1 to 2 to 3
			{
				basis2[v][i][j] = futures_price[v][i][0][j] * futures_price[v][i][0][j];
			}
		}

		//// F_i_j_c* F_i_j*n
		for (int j = 0; j < stage - v; j++) //j from 0 1 to 2 to 3
		{
			basis3[v][j][0] = futures_price[v][0][0][j] * futures_price[v][1][0][j];
			basis3[v][j][1] = futures_price[v][1][0][j] * futures_price[v][2][0][j];
			basis3[v][j][2] = futures_price[v][0][0][j] * futures_price[v][2][0][j];
		}

		// F_i_j*F_i_j+1
		if (v < stage - 1)
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - v - 1; j++) // j from 0 1 to 2
				{
					basis4[v][i][j] = futures_price[v][i][0][j] * futures_price[v][i][0][j + 1]; // basis4 is from 0 to 21
																								 //cout << basis4[v][i][j] << endl;
				}
			}
		}
	}
}

double PenaltyFunction(int action, int state, int current_stage, int nofcoeffs, vector<vector<vector<vector<double>>>> &futures_price, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI)
{
	double s = 0;
	if (current_stage > stage - 1)
		return s;
	else
	{
		if (state == 'O' && (action == 'P' || action == 'S'))
		{
			int position = 1;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * (basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1]);
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * (basis2[current_stage][i][j] - basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1]);
					position++;
				}
			}

			// last basis function
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - current_stage - 1; j++)
					{
						s += coefficient[current_stage][1][position] * (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
				position++;
			}
			return s;
		}


		if (state == 'O' && action == 'M')
		{
			int position = 1;
			// F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * (basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1]);
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * (basis2[current_stage][i][j] - basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1]);
					position++;
				}
			}

			// last basis function
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - current_stage - 1; j++)
					{
						s += coefficient[current_stage][2][position] * (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
				position++;

				s += coefficient[current_stage][2][position] * (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
				position++;

				s += coefficient[current_stage][2][position] * (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
				position++;
			}
			return s;
		}

		if (state == 'M1' && action == 'M')
		{
			int position = 1;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1]);
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis2[current_stage][i][j] - basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1]);
					position++;
				}
			}

			// last basis function
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - current_stage - 1; j++)
					{
						s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
				position++;

				s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
				position++;

				s += coefficient[current_stage][1 + time_lag_mothball][position] * (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
				position++;
			}
			return s;
		}


		if (state == 'M1' && action == 'R')
		{
			int position = 1;
			for (int i = 0; i < 3; i++) // i from 0 to 2
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++) // current stage is 0; j is from 0 to 1
				{
					s += coefficient[current_stage][1][position] * (basis1[current_stage][i][j] - basis1[current_stage - time_lag_reactivation][i][j + time_lag_reactivation]);
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)// current stage is 0; j is from 0 to 1
				{
					double final_loadingcoefficient1 = 1;
					for (int k = 0; k < time_lag_reactivation; k++)
					{
						final_loadingcoefficient1 *= loadingcoeffs1[current_stage - time_lag_reactivation + k][i][0][j + time_lag_reactivation - k];
					}
					s += coefficient[current_stage][1][position] * (basis2[current_stage][i][j] - basis2[current_stage - time_lag_reactivation][i][j + time_lag_reactivation] * final_loadingcoefficient1);
					position++;
				}
			}

			// F_i_j * F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						double final_loadingcoefficient3 = 1;
						for (int k = 0; k < time_lag_reactivation; k++)
						{
							final_loadingcoefficient3 *= loadingcoeffs3[current_stage - time_lag_reactivation + k][i][0][j + time_lag_reactivation - k];
						}
						s += coefficient[current_stage][1][position] * (basis4[current_stage][i][j] - basis4[current_stage - time_lag_reactivation][i][j + time_lag_reactivation] * final_loadingcoefficient3);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				double final_loadingcoefficient2_0 = 1;
				double final_loadingcoefficient2_2 = 1;
				double final_loadingcoefficient2_1 = 1;
				for (int k = 0; k < time_lag_reactivation; k++)
				{
					final_loadingcoefficient2_0 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][0][0][j + time_lag_reactivation - k];
					final_loadingcoefficient2_2 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][2][0][j + time_lag_reactivation - k];
					final_loadingcoefficient2_1 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][1][0][j + time_lag_reactivation - k];
				}
				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][0] - basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][0] * final_loadingcoefficient2_0);
				position++;

				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][2] - basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][2] * final_loadingcoefficient2_2);
				position++;

				s += coefficient[current_stage][1][position] * (basis3[current_stage][j][1] - basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][1] * final_loadingcoefficient2_1);
				position++;

			}
			return s;
		}
	}
}

double greedypolicy(int i, int state, double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI)
{ 
	double v1 = -1000;
	double v2 = -1000;
	double v3 = -1000;
	double v4 = -1000;

	if (i == stage - 1) {
		return salvage_value;
	}
	else {
		switch (state)
		{
		case'O':
		{
			//First action: abandon
			v1 = rewardFunction('O', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);

			//Second action: Produce
			if (i < stage - 1)
			{
				v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('P', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}
			//Third action: mothball
			if (i < stage - 1 - time_lag_reactivation)
			{
				v3 = rewardFunction('O', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('M', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}

			// Forth action: Suspend
			if (i < stage - 1)
			{
				v4 = rewardFunction('O', 'S', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('S', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}

			////////////////////////////////////////////////////
			//Decision
			////////////////////////////////////////////////////
			if ((v1 >= v2 && v1 >= v3 && v1 >= v4) || i == stage - 1) // the action is abandon // in the last stage, only abandon is available
			{
				return v1;
			}

			if (v2 >= v1 && v2 >= v3 && v2 >= v4) // the action is produce
			{
				return v2;
			}

			if (v3 >= v1 && v3 >= v2 && v3 >= v4) // the action is mothball
			{
				return v3;                     // jump to the M2 stage
			}

			if (v4 >= v1 && v4 >= v2 && v4 >= v3) // the action is suspend 
			{
				return v4;
			}
			break;
		}

		case 'M1':
		{
			if (i < stage - 1 - time_lag_reactivation) // reward for action keeping mothballing in state M2, only happen before 20.
			{
				v1 = rewardFunction('M1', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('M', 'M1', i + 1, forwardCurveSample_1, tmpPMI);
			}
			v2 = rewardFunction('M1', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
			v3 = rewardFunction('M1', 'R', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('R', 'M1', i + time_lag_reactivation, forwardCurveSample_1, tmpPMI);

			if (v1 >= v2 && v1 >= v3) // action is M, keep mothballing
			{
				return v1;
			}

			if (v2 >= v1 && v2 >= v3) // action is abandon
			{
				return v2;
			}

			if (v3 >= v2 && v3 >= v1) // action is R, keep mothballing
			{
				return v3;
			}
			break;
		}
		}//END of Switch
	}
}//end of function

double upperbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI)
{
	double V[stage][max_state - 1] = { 0 };

	// 0 is operational
	// 1 is mothballed

	V[stage - 1][0] = salvage_value;

	for (int i = stage - 2; i >= 0; i--)
	{
		for (int j = 0; j < statesForEachStage[i].size(); j++)
		{
			double v1 = -1000;
			double v2 = -1000;
			double v3 = -1000;
			double v4 = -1000;

			switch (statesForEachStage[i][j])
			{
			case 'A':
				break;
			case 'O':
				v1 = rewardFunction('O', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);

				//Second action: Produce
				if (i < stage - 1)
				{
					v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - discount_factor * PenaltyFunction('P', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI) + discount_factor * V[i + 1][0];
				}
				//Third action: Mothball
				if (i < stage - 1 - time_lag_reactivation)
				{
					v3 = rewardFunction('O', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - discount_factor * PenaltyFunction('M', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI) + discount_factor * V[i + 1][1];
				}

				// Forth action: Suspend
				if (i < stage - 1)
				{
					v4 = rewardFunction('O', 'S', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - discount_factor * PenaltyFunction('S', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI) + discount_factor * V[i + 1][0];
				}

				////////////////////////////////////////////////////
				//Decision
				////////////////////////////////////////////////////
				if (v1 >= v2 && v1 >= v3 && v1 >= v4) // the action is abandon // in the last stage, only abandon is available
				{
					V[i][0] = v1;
				}

				if (v2 >= v1 && v2 >= v3 && v2 >= v4) // the action is produce
				{
					V[i][0] = v2;
				}

				if (v3 >= v1 && v3 >= v2 && v3 >= v4) // the action is mothball
				{
					V[i][0] = v3;             
				}

				if (v4 >= v1 && v4 >= v2 && v4 >= v3) // the action is suspend 
				{
					V[i][0] = v4;
				}

				break;
			case 'M1':
				if (i < stage - 1 - time_lag_reactivation)
				{
					v1 = rewardFunction('M1', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - discount_factor * PenaltyFunction('M', 'M1', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI) + discount_factor * V[i + 1][1];
				}
				v2 = rewardFunction('M1', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
				v3 = rewardFunction('M1', 'R', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - discount_factor * PenaltyFunction('R', 'M1', i + time_lag_reactivation, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI) + discount_factor * V[i + time_lag_reactivation][0];

				if (v1 >= v2 && v1 >= v3) // action is M, keep mothballing
				{
					V[i][1] = v1;
				}

				if (v2 >= v1 && v2 >= v3) // action is abandon
				{
					V[i][1] = v2;
				}

				if (v3 >= v2 && v3 >= v1) // action is R, keep mothballing
				{
					V[i][1] = v3;
				}
				break;
			}//endl of switch
		} // end of j
	} // end of i
	return V[0][0];
}

void lowerbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI)
{
	int i = 0;// i is the stage
	while (i < stage)
	{
		double v1 = -1000;
		double v2 = -1000;
		double v3 = -1000;
		double v4 = -1000;

		switch (stateSet[i])
		{
		case 'NULL': // if current state is Abandonment
		{
			for (int j = i; j < stage; j++) // then from current stage i to the last stage 23, all action and state are "Abandonment"
			{
				action[j] = 'NULL';
				stateSet[j] = 'NULL';
			}
			i = stage; // set i to 24 and then end the while-loop;
			break;
		}
		case'O':
		{
			//First action: abandon
			v1 = rewardFunction('O', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);

			//Second action: Produce
			if (i < stage - 1)
			{
				v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('P', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}
			//Third action: Mothball
			if (i < stage - 1 - time_lag_reactivation)
			{
				v3 = rewardFunction('O', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('M', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}

			// Forth action: Suspend
			if (i < stage - 1)
			{
				v4 = rewardFunction('O', 'S', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('S', 'O', i + 1, forwardCurveSample_1, tmpPMI);
			}

			////////////////////////////////////////////////////
			//Decision
			////////////////////////////////////////////////////
			if ((v1 >= v2 && v1 >= v3 && v1 >= v4) || i == stage - 1) // the action is abandon
			{
				action[i] = 'A';
				stateSet[i + 1] = 'NULL';
				i++;
			}

			if (v2 >= v1 && v2 >= v3 && v2 >= v4) // the action is produce
			{
				action[i] = 'P';
				stateSet[i + 1] = 'O';
				i++;
			}

			if (v3 >= v1 && v3 >= v2 && v3 >= v4) // the action is mothball
			{
				action[i] = 'M';
				stateSet[i + 1] = 'M1';
				i++;
			}

			if (v4 >= v1 && v4 >= v2 && v4 >= v3) // the action is suspend 
			{
				action[i] = 'S';
				stateSet[i + 1] = 'O';
				i++;
			}
			break;
		}

		case 'M1':
		{
			if (i < stage - 1 - time_lag_reactivation) // reward for action keeping mothballing in state M2, only happen before 20.
			{
				v1 = rewardFunction('M1', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('M', 'M1', i + 1, forwardCurveSample_1, tmpPMI);
			}
			v2 = rewardFunction('M1', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
			v3 = rewardFunction('M1', 'R', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Continuation_Value('R', 'M1', i + time_lag_reactivation, forwardCurveSample_1, tmpPMI);
			if (v1 >= v2 && v1 >= v3) // action is M, keep mothballing
			{
				action[i] = 'M';
				stateSet[i + 1] = 'M1';
				i++;
			}

			if (v2 >= v1 && v2 >= v3) // action is abandon
			{
				action[i] = 'A';
				stateSet[i + 1] = 'NULL';
				i++;
			}

			if (v3 >= v2 && v3 >= v1) // action is R, keep mothballing
			{
				action[i] = 'R';
				stateSet[i + time_lag_reactivation] = 'O';
				for (int k = 1; k < time_lag_reactivation; k++)
				{
					action[i + k] = 'Empt';
					stateSet[i + k] = 'Empt';
				}
				i = i + time_lag_reactivation;
			}
			break;
		}
		}
	}
}

/********************************************************************************/
//Continuation Value Approximation
/********************************************************************************/
double Continuation_Value(int action, int state, int current_stage, vector<vector<vector<vector<double>>>> &futures_price, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI)
{
	//input current stage is i+1 from 5 to 2
	//input current stage is i+1 from 1 ~ 6. 
	double s = 0;

	if (current_stage > stage - 1)
		return s;
	else
	{
		if (state == 'O' && (action == 'P' || action == 'S'))
		{
			int position = 0;
			s += coefficient[current_stage][1][position]; // constant basis functions
			position++;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++) // when stage is 0, j is from 0 to 21
				{
					s += coefficient[current_stage][1][position] * basis1[current_stage - 1][i][j + 1]; // Here the basis is actually the expected basis
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						s += coefficient[current_stage][1][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1];
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1];
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1];;
				position++;
			}
			return s;
		}

		/*******************************************************************************************************************************************/

		if ((state == 'O' && action == 'M'))
		{
			int position = 0;
			//constant
			s += coefficient[current_stage][2][position];
			
			position++;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++) // when stage is 0, j is from 0 to 21
				{
					s += coefficient[current_stage][2][position] * basis1[current_stage - 1][i][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						s += coefficient[current_stage][2][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1];
				position++;
			}
			return s;
		}

		/*******************************************************************************************************************************************/

		if (state == 'M1' && action == 'M')
		{
			int position = 0;
			s += coefficient[current_stage][2][position]; // constant basis functions
			position++;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basis1[current_stage - 1][i][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						s += coefficient[current_stage][2][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1];
				position++;
			}
			return s;
		}

		//*******************************************************************************************************************************************
		//********************************************************************************************************************************************
		if (state == 'M1' && action == 'R')
		{
			int position = 0;
			s += coefficient[current_stage][1][position];
			position++;

			for (int i = 0; i < 3; i++) // i from 0 to 2
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basis1[current_stage - time_lag_reactivation][i][j + time_lag_reactivation];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					double final_loadingcoefficient1 = 1;
					for (int k = 0; k < time_lag_reactivation; k++)
					{
						final_loadingcoefficient1 *= loadingcoeffs1[current_stage - time_lag_reactivation + k][i][0][j + time_lag_reactivation - k];
					}
					s += coefficient[current_stage][1][position] * basis2[current_stage - time_lag_reactivation][i][j + time_lag_reactivation] * final_loadingcoefficient1;
					position++;
				}
			}

			// F_i_j * F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						double final_loadingcoefficient3 = 1;
						for (int k = 0; k < time_lag_reactivation; k++)
						{
							final_loadingcoefficient3 *= loadingcoeffs3[current_stage - time_lag_reactivation + k][i][0][j + time_lag_reactivation - k];
						}
						s += coefficient[current_stage][1][position] * basis4[current_stage - time_lag_reactivation][i][j + time_lag_reactivation] * final_loadingcoefficient3;
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				double final_loadingcoefficient2_0 = 1;
				double final_loadingcoefficient2_2 = 1;
				double final_loadingcoefficient2_1 = 1;
				for (int k = 0; k < time_lag_reactivation; k++)
				{
					final_loadingcoefficient2_0 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][0][0][j + time_lag_reactivation - k];
					final_loadingcoefficient2_2 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][2][0][j + time_lag_reactivation - k];
					final_loadingcoefficient2_1 *= loadingcoeffs2[current_stage - time_lag_reactivation + k][1][0][j + time_lag_reactivation - k];
				}
				s += coefficient[current_stage][1][position] * basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][0] * final_loadingcoefficient2_0;
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][2] * final_loadingcoefficient2_2;
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - time_lag_reactivation][j + time_lag_reactivation][1] * final_loadingcoefficient2_1;
				position++;
			}
			return s;
		}
	}
}

/**************************************************************************
* Generate matrix_A
**************************************************************************/
void generation_A_matrix(int loop_of_sample_path)
{
	for (int current_stage = 1; current_stage < stage; current_stage++)
	{
		// loop_ stage from 1 to stage-1
		int position = 0;
		matrix_A[current_stage][loop_of_sample_path][position] = 1;
		position++;
		// F_i_j
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < stage - current_stage; j++)
			{
				matrix_A[current_stage][loop_of_sample_path][position] = basis1[current_stage][i][j];
				position++;
			}
		}

		//// F_i_j*F_i_j
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < stage - current_stage; j++)
			{
				matrix_A[current_stage][loop_of_sample_path][position] = basis2[current_stage][i][j];
				position++;
			}
		}

		// F_i_j*F_i_j+1
		if (current_stage < stage - 1)
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - 1 - current_stage; j++)
				{
					matrix_A[current_stage][loop_of_sample_path][position] = basis4[current_stage][i][j];
					position++;
				}
			}
		}

		// F_i_j_c* F_i_j*n
		for (int j = 0; j < stage - current_stage; j++)
		{
			matrix_A[current_stage][loop_of_sample_path][position] = basis3[current_stage][j][0];
			position++;

			matrix_A[current_stage][loop_of_sample_path][position] = basis3[current_stage][j][2];
			position++;

			matrix_A[current_stage][loop_of_sample_path][position] = basis3[current_stage][j][1];
			position++;
		}
	}
}

//**************************************************************************************************
//* Simulates a forward curve sample for each stage using the factor model and antithetic variates
//**************************************************************************************************
inline void simulateForwardCurveAntitheticFM(
	struct_PriceModelInfo<vector<vector<vector<double> > > > &MI,
	vector<vector<vector<vector<double> > > > &forwardCurveSample) {
	int nHedgeStageIndex;
	int totStages = MI.nStages;
	int nTempStages = MI.nStages;
	int nPosition;
	int uIndexCurrentNumber = 0;
	int tmpMonth, maturity, dayShift;
	MI.RndVariable.resize(MI.nRealFactors);
	double timeDelta = MI.timeDelta;
	double uHedgeMultiplier;
	//beginning of simulating the whole price curve
	//
	while (nTempStages)
	{
		tmpMonth = (totStages - nTempStages) / MI.nDaysPerMonth;
		dayShift = (totStages - nTempStages) % MI.nDaysPerMonth;
		nHedgeStageIndex = (MI.nStartingMonth - 1 + tmpMonth) % 12;

		if (nTempStages == totStages) {
			for (int i = 0; i < MI.nCommodities; i++)
			{
				for (int j = 0; j < MI.nStages + 1; j++)
				{
					forwardCurveSample[0][i][0][j] = MI.initForwCurves[i][0][j];
				}
			}
			nPosition = 0;
		}
		else {
			nPosition = totStages - nTempStages;
		}

		//Implementation of antithetic variates
		if (MI.RndControl == 0) {
			for (int j = 0; j < MI.nRealFactors; j++) {
				MI.RndVariable[j] = randn(0, 1, MI.RndVarSeed);
				MI.StoredRndVariable[uIndexCurrentNumber + j] =
					MI.RndVariable[j];
			}
			uIndexCurrentNumber += MI.nRealFactors;
		}
		else {
			for (int j = 0; j < MI.nRealFactors; j++)
				MI.RndVariable[j] = -MI.StoredRndVariable[uIndexCurrentNumber
				+ j];
			uIndexCurrentNumber += MI.nRealFactors;
		}

		// This part evolves the forward curve across the month. Both futures prices and spot prices
		// will change in this calculation.
		for (int i = 0; i < nTempStages; i++) // i is from 0 to stage-2, from 0 to 4
		{
			maturity = (dayShift + i) / MI.nDaysPerMonth;
			for (int c = 0; c < MI.nCommodities; c++) {
				for (int mk = 0; mk < MI.nMarkets; mk++) {
					uHedgeMultiplier = 0;
					for (int j = 0; j < MI.nRealFactors; j++) {
						uHedgeMultiplier +=
							sqrt(timeDelta)
							* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity]
							* MI.RndVariable[j]
							- timeDelta
							* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity]
							* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity]
							/ (double)2;
					}
					forwardCurveSample[nPosition + 1][c][mk][i] =
						forwardCurveSample[nPosition][c][mk][i + 1]
						* exp(uHedgeMultiplier); // i is from 0 to 4
				}
			}
		}


		////////////////////////////////////////////////
		// generate the loading coeffs matrix///////////
		////////////////////////////////////////////////
		if (controlvalue == 0)
		{
			for (int i = 0; i < nTempStages; i++) //i is from 0 to stage-2, 0 to 4
			{
				maturity = (dayShift + i) / MI.nDaysPerMonth;
				for (int c = 0; c < MI.nCommodities; c++)
				{
					for (int mk = 0; mk < MI.nMarkets; mk++)
					{
						uHedgeMultiplier = 0;
						for (int j = 0; j < MI.nRealFactors; j++)
						{
							uHedgeMultiplier +=
								timeDelta
								* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity]
								* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity];
						}

						loadingcoeffs1[nPosition][c][mk][i + 1]
							= exp(uHedgeMultiplier); // i is from 1 to 
					}
				}
			}

			for (int i = 0; i < nTempStages; i++)
			{
				maturity = (dayShift + i) / MI.nDaysPerMonth;
				////////////////////
				uHedgeMultiplier = 0;
				for (int j = 0; j < MI.nRealFactors; j++)
				{
					uHedgeMultiplier +=
						timeDelta
						* MI.loadingCoeffs[0][0][nHedgeStageIndex][j][maturity]
						* MI.loadingCoeffs[1][0][nHedgeStageIndex][j][maturity];
				}

				loadingcoeffs2[nPosition][0][0][i + 1]
					= exp(uHedgeMultiplier);

				//////////////////////////////////////
				uHedgeMultiplier = 0;
				for (int j = 0; j < MI.nRealFactors; j++)
				{
					uHedgeMultiplier +=
						timeDelta
						* MI.loadingCoeffs[1][0][nHedgeStageIndex][j][maturity]
						* MI.loadingCoeffs[2][0][nHedgeStageIndex][j][maturity];
				}

				loadingcoeffs2[nPosition][1][0][i + 1]
					= exp(uHedgeMultiplier);

				//	///////////////////
				uHedgeMultiplier = 0;
				for (int j = 0; j < MI.nRealFactors; j++)
				{
					uHedgeMultiplier +=
						timeDelta
						* MI.loadingCoeffs[0][0][nHedgeStageIndex][j][maturity]
						* MI.loadingCoeffs[2][0][nHedgeStageIndex][j][maturity];
				}
				loadingcoeffs2[nPosition][2][0][i + 1]
					= exp(uHedgeMultiplier);
			}
			//////////////////////////////////
			//////////////////////////////////
			for (int i = 0; i < nTempStages - 1; i++) {
				maturity = (dayShift + i) / MI.nDaysPerMonth;
				for (int c = 0; c < MI.nCommodities; c++) {
					for (int mk = 0; mk < MI.nMarkets; mk++) {
						uHedgeMultiplier = 0;
						for (int j = 0; j < MI.nRealFactors; j++) {
							uHedgeMultiplier +=
								timeDelta
								* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity]
								* MI.loadingCoeffs[c][mk][nHedgeStageIndex][j][maturity + 1]
								;
						}
						loadingcoeffs3[nPosition][c][mk][i + 1]
							= exp(uHedgeMultiplier);
					}
				}
			}
		}
		/////////////////////////////////
		/////////////////////////////////
		nTempStages = nTempStages - 1;
	}
	if (MI.RndControl == 0)
		MI.RndControl = 1;
	else
		MI.RndControl = 0;
	controlvalue = 1;
}

#endif /* MULTCOMMTERMSTRUCTUREMODEL_HPP_ */

double cpp_genalea(int *x0) {
	int m = 2147483647; // 2^31-1
	int a = 16807;  // 7^5
	int b = 127773;
	int c = 2836;
	int x1, k;

	k = (int)((*x0) / b);
	x1 = a * (*x0 - k * b) - k * c;
	if (x1 < 0)
		x1 = x1 + m;
	*x0 = x1;

	return ((double)x1 / (double)m);
}
// normal distribution N(0,1)

double randn(const double mu, const double sigma, int& seed) {
	double polar, rsquared, var1, var2;
	do {
		var1 = 2.0 * (cpp_genalea(&seed)) - 1.0;  // var1 and var2 are between -1 and 1 
		var2 = 2.0 * (cpp_genalea(&seed)) - 1.0; //// var1 and var2 are between -1 and 1 
		rsquared = var1 * var1 + var2 * var2;
	} while (rsquared >= 1.0 || rsquared == 0.0); // so the "rsquared" is between (0, 1)
												  //cout << "generate randn" << endl;
	polar = sqrt(-2.0 * log(rsquared) / rsquared);

	return var2 * polar * sigma + mu;
}

void initializePriceModelStructures(ifstream &priceModelCalibInputFile,
	ifstream &priceModelInitCondnInputFile,
	//struct_cmdLineParams &cmdLineParams,
	vector<vector<vector<vector<double> > > > &forwardCurveSample,
	struct_PriceModelInfo<vector<vector<vector<double> > > > &tmpPMI)

{
	priceModelCalibInputFile >> tmpPMI.loadingCoeffMonths;
	priceModelCalibInputFile >> tmpPMI.loadingCoeffMats;
	priceModelCalibInputFile >> tmpPMI.loadingCoeffComm;
	priceModelCalibInputFile >> tmpPMI.loadingCoeffMkts;

	tmpPMI.loadingCoeffs.resize(tmpPMI.loadingCoeffComm);
	for (int c = 0; c < tmpPMI.loadingCoeffComm; c++) {
		tmpPMI.loadingCoeffs[c].resize(tmpPMI.loadingCoeffMkts);
		for (int mk = 0; mk < tmpPMI.loadingCoeffMkts; mk++) {
			tmpPMI.loadingCoeffs[c][mk].resize(tmpPMI.loadingCoeffMonths);
			for (int t = 0; t < tmpPMI.loadingCoeffMonths; t++) {
				tmpPMI.loadingCoeffs[c][mk][t].resize(tmpPMI.loadingCoeffMats*tmpPMI.loadingCoeffComm);
				for (int f = 0; f < tmpPMI.loadingCoeffMats*tmpPMI.loadingCoeffComm; f++) {
					tmpPMI.loadingCoeffs[c][mk][t][f].resize(
						tmpPMI.loadingCoeffMats);
					for (int mo = 0; mo < tmpPMI.loadingCoeffMats; mo++) {
						priceModelCalibInputFile
							>> tmpPMI.loadingCoeffs[c][mk][t][f][mo];
					}
				}
			}
		}
	}

	int inst_nMonths, inst_nDaysPerMonth, inst_nCommodities, inst_nMarkets;

	priceModelInitCondnInputFile >> tmpPMI.nMonths;
	priceModelInitCondnInputFile >> tmpPMI.nDaysPerMonth;
	priceModelInitCondnInputFile >> tmpPMI.nCommodities;
	priceModelInitCondnInputFile >> tmpPMI.nMarkets;
	priceModelInitCondnInputFile >> tmpPMI.discFactor;
	//tmpPMI.discFactor = 0.999758;
	tmpPMI.nStages = tmpPMI.nMonths;
	if (Month == "Jan") {
		tmpPMI.nStartingMonth = 1;
	}
	else if (Month == "Feb") {
		tmpPMI.nStartingMonth = 2;
	}
	else if (Month == "Mar") {
		tmpPMI.nStartingMonth = 3;
	}
	else if (Month == "Apr") {
		tmpPMI.nStartingMonth = 4;
	}
	else if (Month == "May") {
		tmpPMI.nStartingMonth = 5;
	}
	else if (Month == "Jun") {
		tmpPMI.nStartingMonth = 6;
	}
	else if (Month == "Jul") {
		tmpPMI.nStartingMonth = 7;
	}
	else if (Month == "Aug") {
		tmpPMI.nStartingMonth = 8;
	}
	else if (Month == "Sep") {
		tmpPMI.nStartingMonth = 9;
	}
	else if (Month == "Oct") {
		tmpPMI.nStartingMonth = 10;
	}
	else if (Month == "Nov") {
		tmpPMI.nStartingMonth = 11;
	}
	else if (Month == "Dec") {
		tmpPMI.nStartingMonth = 12;
	}

	tmpPMI.nRealFactors = 8;
	tmpPMI.RndControl = 0;
	tmpPMI.RndVarSeed = 5872345;
	tmpPMI.futForwCurve.resize(tmpPMI.nStages);
	tmpPMI.RndVariable.resize((tmpPMI.nStages + 1) * tmpPMI.nRealFactors);
	tmpPMI.StoredRndVariable.resize((tmpPMI.nStages + 1) * tmpPMI.nRealFactors);
	tmpPMI.timeDelta = 1.0 / ((double)(12.0 * tmpPMI.nDaysPerMonth));

	tmpPMI.initForwCurves.resize(tmpPMI.nCommodities);
	for (int c = 0; c < tmpPMI.nCommodities; c++) {
		tmpPMI.initForwCurves[c].resize(tmpPMI.nMarkets);
		for (int m = 0; m < tmpPMI.nMarkets; m++) {
			tmpPMI.initForwCurves[c][m].resize(tmpPMI.nStages + 1);
			for (int t = 0; t < tmpPMI.nStages + 1; t++) {
				priceModelInitCondnInputFile >> tmpPMI.initForwCurves[c][m][t];
			}
		}
	}

	forwardCurveSample.resize(tmpPMI.nStages + 1);
	for (int t = 0; t < tmpPMI.nStages + 1; t++) {
		forwardCurveSample[t].resize(tmpPMI.nCommodities);
		for (int c = 0; c < tmpPMI.nCommodities; c++) {
			forwardCurveSample[t][c].resize(tmpPMI.nMarkets);
			for (int m = 0; m < tmpPMI.nMarkets; m++) {
				forwardCurveSample[t][c][m].resize(tmpPMI.nStages + 1);
				for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++)
				{
					forwardCurveSample[t][c][m][maturity] = 0;
				}
			}
		}
	}
}
