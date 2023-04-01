////**************************************************************************************
//// Date: 03/31/2023
//// Author: Bo Yang, Selvaprabu Nadarajah, Nicola Secomandi
//// Description: C++ Implementation for Pathwise Optimization for Merchant Energy Production
////              24-stage Ethanol Production: Jan Instance
////              BCD Algorithm with Greedy Block Selection Rule
////              High Order Polynomial Basis Functions
//// Parameter Details: 70,000 Samples for Pathwise Linear Program
////                    500,000 Samples for Unbiased Lower and Dual (Upper) Bounds
////                    7 Blocks for Each BCD Iteration
////                    
////**************************************************************************************

//*********************************************
// Standard Headers
//*********************************************
#include "gurobi_c++.h"
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
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <complex>

//*******************************
//External Libraries
//*******************************
#include"dataanalysis.h"
#include <lapacke.h>
#include <cblas.h>

//******************************************************************************
// Definitions of Initial Forward Structure and Loading coefficient Vectors
//******************************************************************************
using namespace std;
using namespace std::chrono;

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
	int segmentStartingPoint;
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
vector<vector<vector<double>>> inputForwardCurves;

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
static const double salvage_value = 0; // Salvage value for the abandonment
static const int time_lag_mothball = 1; // Time lag for the action "Mothball"
static const int time_lag_reactivation = 1; // Time lag for the action "Reactivate"
static const int time_lag = 0;
//*****************************************
// Parameters for the BCD Algorithm
//*****************************************
const int NumOfSamplePath = 70000; 
// Sample Number for Pathwise Linear Program

const int numofsamples = 100000; 
// Sample Number Unbiased Bound Simulation

const int nofcoeffs = 3 * 3 * (stage - 1) + 3 * (stage - 2); 
// High Polynomial Basis function Number at the Initial Stage

const long long NumOfConstraints = NumOfSamplePath * (4 * (stage - time_lag_reactivation - 1) + 3 * (time_lag_reactivation)+1 + 3 * (stage - time_lag_reactivation - 1 - 1) + 2); 
// Total Constraint Number of PLP

long long Number_Of_Beta = 0; 
// Total Number of Beta Variables (Will Update Dynamically Later)

//***************************************************
// Parameters for the PCA Preconditioning Procedure
//***************************************************
int magnitude = -8;

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

double rewardFunction(int, int, int, double, double, double);
// Return payoff for each feasible action

void InputFunction();
// Apply the PCA-based preconditioning procedure

void SampleTransformation();
// Project a forward curve sample to the space spanned by the preconditioned basis functions

void basisFunction(int, int, vector<vector<vector<vector<double> > > >&);
// Compute basis function values for a forward curve sample

string itos(int i) { stringstream s; s << i; return s.str(); }
// Transfer Int to String

//GRBLinExpr penaltyFunction(int, int, int, int, vector<vector<vector<vector<double>>>> &, struct_PriceModelInfo<vector<vector<vector<double> > > > &, int);

GRBLinExpr penaltyFunction(int, int, int, int, int);
// Return the dual penalty for each action. 
// Note that this function returns a linear expression that contains corresponding Beta variables.

double PenaltyFunction(int, int, int, int,
	vector<vector<vector<vector<double>>>> &,
	vector<vector<vector<double>>> &,
	struct_PriceModelInfo<vector<vector<vector<double> > > > &,
	int);
// Return the dual penalty for each action. 
// Note that this function returns the penalty value for fixed Beta variable values

double PenaltyFunction(int action, int state, int current_stage, int nofcoeffs,
	vector<vector<vector<double>>> &coefficient,
	int TotalLoop);

//GRBLinExpr penaltyexpression(int state, int current_stage);

void BasisSubtraction(int current_stage,
	vector<vector<vector<vector<double>>>> &futures_price,
	struct_PriceModelInfo<vector<vector<vector<double> > > > & MI,
	int TotalLoop);
// Compute the subtraction of a basis function and its expected value
// Note that this function computes the substractions for an entire forward curve sample

void ReducedCost(vector<vector<vector<GRBConstr>>>& ConstraintPointer,
	vector<vector<double>>&Reduced_Cost,
	double**basisSubtraction);
// Compute the reduced cost for each Beta variable.
// The greedy selection rule chooses the Beta variables with the largest reduced costs to update

double upperbound(double discount_factor,
	vector<vector<vector<vector<double>>>> &forwardCurveSample_1,
	vector<vector<vector<double>>> &coefficient,
	struct_PriceModelInfo<vector<vector<vector<double> > > > &,
	double U_value[][max_state - 1]);
// Return a unbiased dual (upper) bound estimate for a forward curve sample

void lowerbound(double discount_factor,
	vector<vector<vector<vector<double>>>> &forwardCurveSample_1,
	vector<vector<vector<double>>> &coefficient,
	struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI,
	int);
// Return a unbiased lower bound estimate for a forward curve sample

double Approximation(int action, int state, int current_stage, int nofcoeffs,
	vector<vector<vector<vector<double>>>> &futures_price,
	vector<vector<vector<double>>> &coefficient,
	struct_PriceModelInfo<vector<vector<vector<double> > > > & MI,
	int);
// Return the value function approximation (VFA) for a particular stage and state
// with given Beta variables and a forward curve sample

void generation_A_matrix(int TotalLoop, double*** matrix_A);
// Store a coefficient matrix for the regression that is used to obtain VFAs for the unbiased lower bound 
// (Please see the regression before \S 5 for details)

//void Beta_Matrix_Construction(int, int, int, int, 
//	vector<vector<vector<vector<double>>>> &, 
//	struct_PriceModelInfo<vector<vector<vector<double> > > > &, 
//	int, int);

void Generate_Beta_Matrix(double* results, double *Beta_Matrix, int i);
// Store the coefficient matrix for the Beta variables in PLP

void Beta_Transformation(vector<vector<vector<double>>> &coefficient);
// Project the Beta variables from the preconditioned space to the original space

void Beta_Update(int i, vector<vector<double>> &beta_var);
// 

void inverse(double* A, int N);
// Inverse the input matrix A

void Inverse_EgenMatrix();
// Inverse the matrix consisting of eigen vectors 

void Initialization(vector<vector<vector<vector<double>>>>forwardCurveSample_0,
	vector<vector<vector<vector<double>>>> forwardCurveSample,
	double* discount_factor, int* tmpPMI_nStages, int*tmpPMI_nMarkets, int*tmpPMI_nCommodities);
// Read initial forward curve and loading coefficients from files

void Order_Magnitude(double *results, int i, int *scale, int *scale_R);
// Rescale the coefficient matrix of the Beta variables after the PCA
// Note that this step does not influence the final results

//***********************************
// Definitions of Variable Vectors
//***********************************
vector<vector<vector<double>>> coefficient; // Store the solution of Beta variables in the PLP
vector<vector<vector<double>>> VFA_coefficient; // Store the VFA used for the unbiased lower bound
double *****forwardcurve_Total = new double****[NumOfSamplePath]; // Store all forward curve samples
GRBVar value_vars[NumOfSamplePath][stage][max_state]; // U variables in PLP
GRBVar coeff_vars[stage][max_state][nofcoeffs]; // Beta variables in PLP
GRBVar Value_vars[stage][max_state]; // U variables used for unbiased dual (upper) bound
int zeroposition[nofcoeffs + 1] = { 0 }; // Vector to store the position of the null column in the post PCA coefficient matrix

static int action[stage] = { 0 };
// Store the feasible action for each stage

static int stateSet[stage] = { 0 };
// Store the feasible endogenous states for each stage

static double greedy_policy_value[stage] = { 0 };
// Store the greedy policy value at each stage for a sequence of actions

double **basisSubtraction = new double*[stage];
// Store the coefficient matrix of Beta variables for each stage 

double **EgenVectorMatrix = new double *[stage];
// Store the eigen vector matrix

double **Inv_EgenVectorMatrix = new double *[stage];
// Store the inverse eigen vector matrix

vector<vector<int>> statesForEachStage;
// Store the endogenous states for each stage.

//************************************************************************
// Definitions of Basis Function Vectors and Loading Coefficient Vectors
//************************************************************************
static double basis1[stage][3][stage] = { 0 };
// Basis functions F_{i,j} for NG, corn, and ethanol 

static double basis2[stage][3][stage] = { 0 };
// Basis functions F_{i,j}^{2} for NG, corn, and ethanol

static double basis3[stage][stage][3] = { 0 };
// Basis functions F_{i,j}^{Corn} * F_{i,i+1}^{NG}, F_{i,j}^{Corn} * F_{i,i+1}^{E}, and F_{i,j}^{E} * F_{i,i+1}^{NG}

static double basis4[stage - 1][3][stage - 1] = { 0 };
// Basis functions F_{i,j} * F_{i,j+1} for NG, corn, and ethanol 

double loadingcoeffs1[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j}^{2} 

double loadingcoeffs2[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j}^{Corn} * F_{i,i+1}^{NG}, F_{i,j}^{Corn} * F_{i,i+1}^{E}, and F_{i,j}^{E}

double loadingcoeffs3[23][3][1][24] = { 0 };
// Loading coefficient vectors for the basis function F_{i,j} * F_{i,j+1} for NG, corn, and ethanol 

//**************************
// Definitions of Variables
//**************************
double AVERAGE = 0;
double diff_Upper_Bound = 0;
double diff_Lower_Bound = 0;
double variance_UpperBound = 0;
double variance_LowerBound = 0;
static double lower_BD = 0;

double PenaltyFunction(int action, int state, int current_stage, int nofcoeffs, vector<vector<vector<double>>> &coefficient, int TotalLoop);

int main(int argc, char *argv[])
{
	try
	{
		int segmentStartingPoint = 0;
		auto start_time = high_resolution_clock::now();

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

		for (int i = 1; i < stage; i++)
		{
			EgenVectorMatrix[i] = new double[(nofcoeffs - 12 * (i - 1))*(nofcoeffs - 12 * (i - 1))];
			Inv_EgenVectorMatrix[i] = new double[(nofcoeffs - 12 * (i - 1))*(nofcoeffs - 12 * (i - 1))];
		}

		for (int i = 0; i < stage; i++)
		{
			basisSubtraction[i] = new double[NumOfSamplePath*(nofcoeffs - 12 * (i - 1))]();
		}

		for (int i = 1; i < stage; i++) // i is from 1 to stage stage-1
		{
			if (i < stage - time_lag_reactivation)
			{
				for (int k = 0; k < max_state - 1; k++) // two states needs approximation. 
				{
					Number_Of_Beta += 3 * 3 * (stage - i) + 3 * (stage - i - 1);
				}
			}
			else
			{
				for (int k = 1; k < max_state - 1; k++) // two states needs approximation. 
				{
					Number_Of_Beta += 3 * 3 * (stage - i) + 3 * (stage - i - 1);
				}
			}
		}

		double *RHS_b = new double[NumOfConstraints];

		vector<vector<vector<vector<double> > > > forwardCurveSample_0; // forward curve
		vector<vector<vector<vector<double> > > > forwardCurveSample;

		double discount_factor = 0;
		int tmpPMI_nStages = stage - 1;
		int tmpPMI_nMarkets = 1;
		int tmpPMI_nCommodities = 3;

		Initialization(forwardCurveSample_0, forwardCurveSample, &discount_factor, &tmpPMI_nStages, &tmpPMI_nMarkets, &tmpPMI_nCommodities);

		coefficient.resize(stage);
		for (int i = 0; i < stage; i++) //NOTE :: we use coefficients starting from stage 1. 
		{
			coefficient[i].resize(max_state); // states needed approximation
			for (int j = 0; j < max_state; j++)
			{
				coefficient[i][j].resize(3 * 3 * (stage - i) + 3 * (stage - i - 1) + 1); // note we insert a constant into each of the coefficient. 
				for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1) + 1; k++)
				{
					coefficient[i][j][k] = 0;
				}
			}
		}

		//string solutionPath = "./Greedy_PO_No_LSM_Beta_Coefficient_" + Month + ".txt";
		//ifstream solution(solutionPath, ios::in);
		//for (int i = 0; i < stage; i++) //NOTE :: we use coefficients starting from stage 1. 
		//{
		//	for (int j = 0; j < max_state; j++)
		//	{
		//		for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1) + 1; k++)
		//		{
		//			solution >> coefficient[i][j][k];
		//		}
		//	}
		//}
		//solution.close();

		cout << "Generating the pathwise linear program (PLP)..." << endl;
		auto LP_start_time = high_resolution_clock::now();
		GRBEnv env = GRBEnv();
		GRBModel *model = new GRBModel(env);
		(*model).getEnv().set(GRB_IntParam_Method, 2);
		(*model).getEnv().set(GRB_IntParam_Crossover, 0);

		for (int k = 0; k < NumOfSamplePath; k++)
		{
			for (int i = 0; i < stage; i++) // i from 0 to 23
			{
				for (int j = 0; j < max_state; j++) // j from 0 to 5
				{
					switch (j)
					{
					case 0:   // each stage has 6 variables corresponding to the continuation function, although many of the variables may never be used. 
					{
						break;
					}
					case 1:
					{
						string s = "U_O_" + itos(i) + "_" + itos(k);
						value_vars[k][i][j] = (*model).addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, s);
						break;
					}
					case 2:
					{
						string s = "U_M1_" + itos(i) + "_" + itos(k);
						value_vars[k][i][j] = (*model).addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, s);
						//(*model).update();
						break;
					}
					}
				}
			}
		}

		(*model).update();

		for (int i = 1; i < stage; i++) // i is from 0 to stage
		{
			for (int k = 0; k < max_state; k++) // three states needs approximation. 
			{
				for (int j = 0; j < 3 * 3 * (stage - i) + 3 * (stage - i - 1); j++) // PO (*model), do not constain constant i nthe basis functions 
				{
					string s = "C_" + itos(i) + "_" + itos(k) + "_" + itos(j);
					coeff_vars[i][k][j] = (*model).addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, s);
				}
			}
		}

		// Integrate variables into (*model)
		(*model).update();

		// objective function is to minimize U_O_0, which is the value function for stage 0 and state "Operation"
		GRBLinExpr objective = 0;

		for (int i = 0; i < NumOfSamplePath; i++)
		{
			objective += value_vars[i][0][1];
		}

		(*model).setObjective(objective / NumOfSamplePath, GRB_MINIMIZE);

		vector<vector<vector<GRBConstr>>> ConstraintPointer;
		ConstraintPointer.resize(stage);
		for (int i = 0; i < stage; i++) {
			ConstraintPointer[i].resize(max_state);
		}

		int NumOfRow = 0;
		cout << "Adding constraints..." << endl;
		for (int i = 1; i < stage; i++) // i is stage. We focus on the beta. So i starts from 1 and ends in the last stage but doesn't contain the last constraint.
		{
			cout << "Stage: " << i << endl;
			for (int j = 0; j < max_state; j++) // j is from 0 to 2
			{
				switch (statesForEachStage[i][j])
				{
					// The constraints should contain O and M from the 1st stage to the last stage.
					// which corresponds to the beta matrix 1-1, 1-2 ... until 23-1.
					// After states O and M, consider state A.
				case 'A':
					break;

				case 'O':
					for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
					{
						string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);

						ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'P', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - penaltyFunction('P', 'O', i, nofcoeffs, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_P"));
						//ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'P', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - PenaltyFunction('P', 'O', i, nofcoeffs, coefficient, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_P"));
						RHS_b[NumOfRow] = (rewardFunction('O', 'P', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
						NumOfRow++;
					}

					for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
					{
						string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);

						ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'S', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - penaltyFunction('S', 'O', i, nofcoeffs, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_S"));
						//ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'S', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - PenaltyFunction('S', 'O', i, nofcoeffs, coefficient, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_S"));
						RHS_b[NumOfRow] = (rewardFunction('O', 'S', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
						NumOfRow++;
					}

					if (i > time_lag_reactivation + time_lag)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);

							ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - time_lag_reactivation][2] >= rewardFunction('M1', 'R', i - time_lag_reactivation, forwardcurve_Total[TotalLoop][i - time_lag_reactivation][0][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][1][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][2][0][0]) - penaltyFunction('R', 'M1', i, nofcoeffs, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_R"));
							//ConstraintPointer[i][1].push_back((*model).addConstr(value_vars[TotalLoop][i - time_lag_reactivation][2] >= rewardFunction('M1', 'R', i - time_lag_reactivation, forwardcurve_Total[TotalLoop][i - time_lag_reactivation][0][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][1][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][2][0][0]) - PenaltyFunction('R', 'M1', i, nofcoeffs, coefficient, TotalLoop) + discount_factor * value_vars[TotalLoop][i][1], s + "_R"));
							RHS_b[NumOfRow] = (rewardFunction('M1', 'R', i - time_lag_reactivation, forwardcurve_Total[TotalLoop][i - time_lag_reactivation][0][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][1][0][0], forwardcurve_Total[TotalLoop][i - time_lag_reactivation][2][0][0]));
							NumOfRow++;
						}
					}

					if (i <= time_lag_reactivation + time_lag)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);

							(*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'A', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]), s + "_A");
							RHS_b[NumOfRow] = (rewardFunction('O', 'A', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
							NumOfRow++;
						}
					}
					break;

				case 'M1':
					for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
					{
						string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);

						ConstraintPointer[i][2].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - penaltyFunction('M', 'O', i, nofcoeffs, TotalLoop) + discount_factor * value_vars[TotalLoop][i][2], s + "_M"));
						//ConstraintPointer[i][2].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][1] >= rewardFunction('O', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - PenaltyFunction('M', 'O', i, nofcoeffs, coefficient, TotalLoop) + discount_factor * value_vars[TotalLoop][i][2], s + "_M"));
						RHS_b[NumOfRow] = (rewardFunction('O', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
						NumOfRow++;
					}

					if (i > time_lag_mothball)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);
							ConstraintPointer[i][2].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][2] >= rewardFunction('M1', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - penaltyFunction('M', 'M1', i, nofcoeffs, TotalLoop) + discount_factor * value_vars[TotalLoop][i][2], s + "_M"));
							//ConstraintPointer[i][2].push_back((*model).addConstr(value_vars[TotalLoop][i - 1][2] >= rewardFunction('M1', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]) - PenaltyFunction('M', 'M1', i, nofcoeffs, coefficient, TotalLoop) + discount_factor * value_vars[TotalLoop][i][2], s + "_M"));
							RHS_b[NumOfRow] = (rewardFunction('M1', 'M', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
							NumOfRow++;
						}
					}

					if (i > time_lag_mothball && i < stage - time_lag_reactivation)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);
							(*model).addConstr(value_vars[TotalLoop][i - 1][2] >= rewardFunction('M1', 'A', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]), s + "_A");
							RHS_b[NumOfRow] = (rewardFunction('M1', 'A', i - 1, forwardcurve_Total[TotalLoop][i - 1][0][0][0], forwardcurve_Total[TotalLoop][i - 1][1][0][0], forwardcurve_Total[TotalLoop][i - 1][2][0][0]));
							NumOfRow++;
						}
					}

					if (i <= time_lag_mothball)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);
							(*model).addConstr(value_vars[TotalLoop][stage - time_lag_reactivation - 1][2] >= rewardFunction('M1', 'A', stage - time_lag_reactivation - 1, forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][0][0][0], forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][1][0][0], forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][2][0][0]), s + "_A");
							RHS_b[NumOfRow] = (rewardFunction('M1', 'A', stage - time_lag_reactivation - 1, forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][0][0][0], forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][1][0][0], forwardcurve_Total[TotalLoop][stage - time_lag_reactivation - 1][2][0][0]));
							NumOfRow++;

							(*model).addConstr(value_vars[TotalLoop][stage - 1][1] == salvage_value, "Last1_" + itos(TotalLoop));
							RHS_b[NumOfRow] = 0;
							NumOfRow++;
						}
					}
					break;
				}
			}
		}

		// Note RHS_b and NumOfRow do not contain the following constraints!
		// 
		for (int i = 1; i < stage - 1; i++) // i is stage. We focus on the beta. So i starts from 1 and ends in the last stage but doesn't contain the last constraint.
		{
			for (int j = 0; j < max_state; j++) // j is from 0 to 2
			{
				switch (statesForEachStage[i][j])
				{
					// The constraints should contain O and M from the 1st stage to the last stage.
					// which corresponds to the beta matrix 1-1, 1-2 ... until 23-1.
					// After states O and M, consider state A.
				case 'A':
					if (i >= time_lag_reactivation)
					{
						for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
						{
							string s = "V_" + itos(i) + "_" + itos(j) + "_" + itos(TotalLoop);
							(*model).addConstr(value_vars[TotalLoop][i][1] >= rewardFunction('O', 'A', i, forwardcurve_Total[TotalLoop][i][0][0][0], forwardcurve_Total[TotalLoop][i][1][0][0], forwardcurve_Total[TotalLoop][i][2][0][0]), s + "_A");
							RHS_b[NumOfRow] = (rewardFunction('O', 'A', i, forwardcurve_Total[TotalLoop][i][0][0][0], forwardcurve_Total[TotalLoop][i][1][0][0], forwardcurve_Total[TotalLoop][i][2][0][0]));
							NumOfRow++;
						}
					}
					break;
				case 'O':
					break;
				case 'M1':
					break;
				}
			}
		}

		(*model).update();
		//(*model).optimize();
		//(*model).write("model.lp");
		cout << "Delete redundant matrix and vectors..." << endl;

		for (int i = 0; i < NumOfSamplePath; i++)
		{
			for (int j = 0; j < tmpPMI_nStages + 1; j++)
			{
				for (int k = 0; k < tmpPMI_nCommodities; k++)
				{
					for (int l = 0; l < tmpPMI_nMarkets; l++)
					{
						delete[] forwardcurve_Total[i][j][k][l];
					}
					delete[] forwardcurve_Total[i][j][k];
				}
				delete[] forwardcurve_Total[i][j];
			}
			delete[] forwardcurve_Total[i];
		}
		delete[] forwardcurve_Total;

		GRBColumn col[stage][max_state][nofcoeffs];

		for (int i = 1; i < stage; i++) // i is from 0 to stage
		{
			for (int k = 0; k < max_state; k++) // three states needs approximation. 
			{
				for (int j = 0; j < 3 * 3 * (stage - i) + 3 * (stage - i - 1); j++) // PO (*model), do not constain constant i nthe basis functions 
				{
					col[i][k][j] = (*model).getCol(coeff_vars[i][k][j]);
				}
			}
		}

		cout << "Remove beta from PLP ..." << endl;
		for (int i = 1; i < stage; i++) // i is from 0 to stage
		{
			for (int k = 0; k < max_state; k++) // three states needs approximation. 
			{
				for (int j = 0; j < 3 * 3 * (stage - i) + 3 * (stage - i - 1); j++) // PO (*model), do not constain constant i nthe basis functions 
				{
					(*model).remove(coeff_vars[i][k][j]);
				}
			}
		}

		(*model).update();
		
		double alpha = 1;
		double negative_alpha = -1;
		double beta = 0;
		int INCX = 1;
		int INCY = 1;
		int Row_RHS = NumOfConstraints;

		double coeff = 0;

		int iteration = 1;
		int block_starting_stage = 1;
		int block_ending_stage = 7;
		int block_stage = 6;
		double Previous_Obj = 0;
		double Current_Obj = 0;
		auto PO_start_time = high_resolution_clock::now();
		vector<double> Penalty(NumOfSamplePath, 0);
		vector<double> Zero_Penalty(NumOfSamplePath, 0);
		string SequentialMinimizationPath = "./PO_BCD_ObjFunValue_Greedy_" + Month + ".txt";
		ofstream SequentialMinimization(SequentialMinimizationPath, ios::out);
		//*****************************************************
		//					Iteration Start
		//*****************************************************

		//****************************************************
		// Greedy Selection Rule
		//****************************************************
		int Max_Block = 14;
		vector<vector<double>> Reduced_Cost;
		Reduced_Cost.resize(stage);
		for (int i = 0; i < stage; i++) {
			Reduced_Cost[i].resize(max_state, 0);
		}

		vector<vector<int>> Block_Selected_Stage_State;
		Block_Selected_Stage_State.resize(stage);
		for (int i = 0; i < stage; i++) {
			Block_Selected_Stage_State[i].resize(max_state, 0);
		}

		vector<vector<int>> Block_Selected_Stage_State_Historical;
		Block_Selected_Stage_State_Historical.resize(stage);
		for (int i = 0; i < stage; i++) {
			Block_Selected_Stage_State_Historical[i].resize(max_state, 0);
		}

		cout << "Performing BCD..." << endl;
		do
		{
			vector<int> Block_Selected;

			//****************************************************************
			// Greedy Block Selectoin Rule
			//****************************************************************
			if (iteration == 1) {
				int count = 0;
				for (int i = 1; i < stage; i++) {
					for (int j = 1; j < max_state; j++) {
						if (count <= Max_Block) {
							Block_Selected_Stage_State[i][j] = 1;
							count++;
						}
					}
				}
			}
			else {
				ReducedCost(ConstraintPointer, Reduced_Cost, basisSubtraction);

				vector<double> temp_reduced_cost;
				for (int i = 1; i < stage; i++) {
					for (int j = 1; j < max_state; j++) {
						if (!(i == stage - 1 && j == max_state - 1)) {
							temp_reduced_cost.push_back(Reduced_Cost[i][j]);
						}
					}
				}

				sort(temp_reduced_cost.begin(), temp_reduced_cost.end());
				double threshold = temp_reduced_cost[temp_reduced_cost.size() - Max_Block];

				for (int i = 1; i < stage; i++) {
					for (int j = 0; j < max_state; j++) {
						if (Reduced_Cost[i][j] >= threshold) {
							Block_Selected_Stage_State[i][j] = 1;
							Block_Selected_Stage_State_Historical[i][j]++;
						}
					}
				}
			}

			//*****************************************************************
			// BCD Iteration
			//*****************************************************************
			//if (block_starting_stage == 1) {
			Previous_Obj = Current_Obj;
			//}

			for (int i = 1; i < stage; i++) { // i is stage. We focus on the beta. So i starts from 1 and ends in the last stage but doesn't contain the last constraint.
				for (int j = 0; j < max_state; j++) { // j is from 0 to 2
					if (Block_Selected_Stage_State[i][j] == 1) {
						for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1); k++) {
							coefficient[i][j][k + 1] = 0;
						}
					}
				}
			}

			vector<double> Abeta;
			// Update RHS
			for (int i = 1; i < stage; i++) {
				int mm = NumOfSamplePath;
				int nn = 1;
				int kk = nofcoeffs - 12 * (i - 1);
				int LDA = mm;
				int LDB = kk;
				int LDC = mm;

				if (i < stage - 1) {
					//State O
					cblas_dgemv(CblasColMajor, CblasNoTrans, mm, kk, negative_alpha, basisSubtraction[i], LDA, &coefficient[i][1][1], INCX, beta, &Penalty[0], INCY);
					//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mm, nn, kk, negative_alpha, basisSubtraction[i], LDA, &coefficient[i][1][1], LDB, beta, &Penalty[0], LDC);
					if (i <= time_lag_reactivation + time_lag) {
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Zero_Penalty.begin(), Zero_Penalty.end());
					}
					else {
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
					}
					//State M
					cblas_dgemv(CblasColMajor, CblasNoTrans, mm, kk, negative_alpha, basisSubtraction[i], LDA, &coefficient[i][2][1], INCX, beta, &Penalty[0], INCY);
					//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mm, nn, kk, negative_alpha, basisSubtraction[i], LDA, &coefficient[i][2][1], LDB, beta, &Penalty[0], LDC);
					if (i == 1) {
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Zero_Penalty.begin(), Zero_Penalty.end());
						Abeta.insert(Abeta.end(), Zero_Penalty.begin(), Zero_Penalty.end());
					}
					else {
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
						Abeta.insert(Abeta.end(), Zero_Penalty.begin(), Zero_Penalty.end());
					}
				}
				else {//last stage
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mm, nn, kk, negative_alpha, basisSubtraction[i], LDA, &coefficient[i][1][1], LDB, beta, &Penalty[0], LDC);
					Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
					Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
					Abeta.insert(Abeta.end(), Penalty.begin(), Penalty.end());
				}
			}

			for (int i = time_lag_reactivation + time_lag; i < stage - 1; i++) {
				Abeta.insert(Abeta.end(), Zero_Penalty.begin(), Zero_Penalty.end());
			}

			//Update RHS
			cblas_daxpy(Row_RHS, alpha, RHS_b, INCX, &Abeta[0], INCY);

			(*model).set(GRB_DoubleAttr_RHS, (*model).getConstrs(), &Abeta[0], (*model).get(GRB_IntAttr_NumConstrs));

			for (int i = 1; i < stage; i++) {
				for (int j = 1; j < max_state; j++) {
					if (Block_Selected_Stage_State[i][j] == 1) {
						for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1); k++) {
							// Note, the current_stage here is beta's stage.
							coeff_vars[i][j][k] = (*model).addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, col[i][j][k], "C_" + itos(i) + "_" + itos(j) + "_" + itos(k));
						}
					}
				}
			}

			(*model).update();
			(*model).optimize();
			Current_Obj = (*model).get(GRB_DoubleAttr_ObjVal);
			cout << "OBJ: " << (*model).get(GRB_DoubleAttr_ObjVal) << endl;
			SequentialMinimization << (*model).get(GRB_DoubleAttr_ObjVal) << endl;

			for (int i = 1; i < stage; i++) {
				for (int j = 1; j < max_state; j++) {
					if (Block_Selected_Stage_State[i][j] == 1) {
						for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1); k++) {
							coefficient[i][j][k + 1] = coeff_vars[i][j][k].get(GRB_DoubleAttr_X);
							(*model).remove(coeff_vars[i][j][k]);
						}
					}
				}
			}

			for (int i = 1; i < stage; i++) {
				for (int j = 1; j < max_state; j++) {
					if (Block_Selected_Stage_State[i][j] == 1) {
						Block_Selected_Stage_State[i][j] = 0;
					}
				}
			}
			iteration++;
		} while (fabs(Current_Obj - Previous_Obj) > 1e-3);
		auto PO_stop_time = high_resolution_clock::now();

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		double unbiased_upperbound = 0;

		string solutionPath = "./Greedy_PO_Beta_Coefficient_" + Month + ".txt";
		ofstream solution(solutionPath, ios::out);
		for (int i = 0; i < stage; i++) //NOTE :: we use coefficients starting from stage 1. 
		{
			for (int j = 0; j < max_state; j++)
			{
				for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - i - 1) + 1; k++)
				{
					solution << setprecision(16) << coefficient[i][j][k] << endl;
				}
			}
		}
		solution.close();

		struct_PriceModelInfo <vector<vector<vector<double>>>> tmpPMI;
		string priceModelInitCondnInputFilePath = "/home/boy1/Ethanol_InputFile/priceModelInitCondnInputFile_" + Month + ".txt";
		ifstream priceModelCalibInputFile_0("/home/boy1/Ethanol_InputFile/priceModelCalibInputFile.txt", ios::in);
		ifstream priceModelInitCondnInputFile_0(priceModelInitCondnInputFilePath, ios::in);

		initializePriceModelStructures(priceModelCalibInputFile_0,
			priceModelInitCondnInputFile_0,
			forwardCurveSample_0,
			tmpPMI);

		priceModelCalibInputFile_0.close();
		priceModelInitCondnInputFile_0.close();

		vector<vector<vector<vector<double> > > > forwardCurveSample_1; // forward curve
		double *sample_Upper_Bound_Value = new double[numofsamples];
		double *sample_Lower_Bound_Value = new double[numofsamples];
		double *matrix_a = new double[(nofcoeffs + 1)*NumOfSamplePath];
		double *b = new double[NumOfSamplePath];

		double ***matrix_A = new double**[stage];
		for (int i = 0; i < stage; i++)
		{
			matrix_A[i] = new double*[NumOfSamplePath];
			for (int j = 0; j < NumOfSamplePath; j++)
			{
				matrix_A[i][j] = new double[nofcoeffs + 1];
				for (int k = 0; k < nofcoeffs + 1; k++)
				{
					matrix_A[i][j][k] = 0;
				}
			}
		}

		double ****matrix_b = new double***[stage];
		for (int i = 0; i < stage; i++)
		{
			matrix_b[i] = new double**[max_state];
			for (int j = 0; j < max_state; j++)
			{
				matrix_b[i][j] = new double*[NumOfSamplePath];
				for (int k = 0; k < NumOfSamplePath; k++)
				{
					matrix_b[i][j][k] = new double[2];
					for (int l = 0; l < 2; l++)
					{
						matrix_b[i][j][k][l] = 0;
					}
				}
			}
		}

		//**********************
		// Recover U variables
		//**********************
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
		
		std::cout << "Generating matrix_A and matrix_b to recover the VFA" << endl;
		tmpPMI.RndVarSeed = 5872345; // Make sure the random seed is identical to the one used for training set
		
		for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++) {

			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample_1);

			// Generate basis functions based on the given sample path
			basisFunction(0, stage, forwardCurveSample_1);

			// Generate matrix A
			generation_A_matrix(TotalLoop, matrix_A);

			// state O: 0-P, 1-S, 2-M, 3-A
			for (int i = 1; i < stage; i++) {
				matrix_b[i][1][TotalLoop][0] = value_vars[TotalLoop][i][1].get(GRB_DoubleAttr_X);
				matrix_b[i][2][TotalLoop][0] = value_vars[TotalLoop][i][2].get(GRB_DoubleAttr_X);
			}
		}// END OF TotalLOOP

		delete model;
		delete[] RHS_b;

		// Transform the P2LP beta into the original space.
		Beta_Transformation(coefficient);

		//	Model_Appro.update();
		auto Upper_Bound_Start_Time = high_resolution_clock::now();
		tmpPMI.RndVarSeed = 6346538;
		for (int TotalLoop = 0; TotalLoop < numofsamples; TotalLoop++)
		{
			double OBJ = 0;
			// start from Jan. Generate sample paths. loop.
			double U_value[stage][max_state - 1] = { 0 };
			// objective function is to minimize U_O_0, which is the value function for stage 0 and state "Operation"

			cout << TotalLoop << endl;

			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample_1);

			int current_stage = 0;

			// generate basis functions based on the given sample path

			basisFunction(current_stage, stage, forwardCurveSample_1);

			for (int current_stage = 1; current_stage < stage; current_stage++)
			{
				BasisSubtraction(current_stage, forwardCurveSample_1, tmpPMI, 0);
			}

			//SampleTransformation();

			//BasisSubtraction_PCA();

			//generation_A_matrix(TotalLoop, matrix_A);

			OBJ = upperbound(tmpPMI.discFactor, forwardCurveSample_1, coefficient, tmpPMI, U_value);
			//add constraints for every state in every stage and for every action.

			unbiased_upperbound += OBJ;

			sample_Upper_Bound_Value[TotalLoop] = OBJ;

		}// END OF TotalLOOP

		cout << "Unbiased upper bound is " << unbiased_upperbound / numofsamples << endl;
		auto Upper_Bound_End_Time = high_resolution_clock::now();

		auto UB_duration = duration_cast<seconds>(Upper_Bound_End_Time - Upper_Bound_Start_Time);
		cout << "Time for estimating the UB: " << UB_duration.count() << endl;

		//end_time_simulation_UB = clock();
		/***********************************************/
		//// Generate the lower bound
		/************************************************/
		//First step, generate the VFA for each state.
		//initialization
		VFA_coefficient.resize(stage);
		for (int i = 0; i < stage; i++)
		{
			VFA_coefficient[i].resize(max_state);
			for (int j = 0; j < max_state; j++)
			{
				VFA_coefficient[i][j].resize(nofcoeffs + 1); //num. of coefficients is nofcoeffs+1, which includes a constant. 
				for (int k = 0; k < nofcoeffs + 1; k++)
				{
					VFA_coefficient[i][j][k] = 0;
				}
			}
		}

		for (int sstage = 0; sstage < stage; sstage++)
		{
			for (int k = 0; k < nofcoeffs; k++)
			{
				for (int kk = k + 1; kk < nofcoeffs + 1; kk++)
				{
					if (matrix_A[sstage][0][k] == matrix_A[sstage][0][kk] && matrix_A[sstage][0][k] != 0)
					{
						for (int loop = 0; loop < NumOfSamplePath; loop++)
						{
							matrix_A[sstage][0][kk] = 0;
						}
					}
				}
			}
		}

		// regression against point estimate upper bound 
		// regression against point estimate upper bound 
		auto Lower_Bound_Start_Time = high_resolution_clock::now();
		//cout <<"Performing least squares regression to obtain VFAs for the lower bound..."<<endl;
		//tmpPMI.RndVarSeed = 5384751;
		for (int i = 1; i < stage; i++) {
			for (int j = 0; j < max_state; j++) {
				lapack_int M = NumOfSamplePath;
				lapack_int N = 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1;
				lapack_int LDA = M;
				lapack_int LDB = M;
				lapack_int RHSN = 1;
				int NumofColinearity;
				char TRANS = 'N';

				int count = 0;
				for (int k = 0; k <3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++) {
					if (matrix_A[i][0][k] != 0) {
						for (int loop = 0; loop < NumOfSamplePath; loop++) {
							matrix_a[count] = matrix_A[i][loop][k];
							count++;
						}
					}
					else {
						zeroposition[k] = 1;
						N = N - 1;
					}
				}

				switch (statesForEachStage[i][j])
				{
				case 'O':
					for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++) {
						b[TotalLoop] = matrix_b[i][1][TotalLoop][0];
						//cout << TotalLoop << endl;
						//cout << b[TotalLoop] << endl;
					}

					//dgels_(&TRANS, &M, &N, &RHSN, matrix_a, &M, b, &M, &work, &lwork, &info);
					LAPACKE_dgels(LAPACK_COL_MAJOR, TRANS, M, N, RHSN, matrix_a, M, b, M);
					//cout << info << endl;
					NumofColinearity = 0;
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++)
					{
						if (zeroposition[k] != 0)
						{
							NumofColinearity++;
							VFA_coefficient[i][1][k] = 0;
						}
						else
						{
							VFA_coefficient[i][1][k] = b[k - NumofColinearity];
						}
					}

					for (int k = 0; k < nofcoeffs + 1; k++)
					{
						zeroposition[k] = 0;
					}
					break;

				case 'M1':
					for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
					{
						b[TotalLoop] = matrix_b[i][2][TotalLoop][0];
						//cout << b[TotalLoop] << endl;
					}
					LAPACKE_dgels(LAPACK_COL_MAJOR, TRANS, M, N, RHSN, matrix_a, M, b, M);
					//dgels_(&TRANS, &M, &N, &RHSN, matrix_a, &M, b, &M, &work, &lwork, &info);
					//cout << info << endl;
					NumofColinearity = 0;
					for (int k = 0; k < 3 * 3 * (stage - i) + 3 * (stage - 1 - i) + 1; k++)
					{
						if (zeroposition[k] != 0)
						{
							NumofColinearity++;
							VFA_coefficient[i][2][k] = 0;
						}
						else
						{
							VFA_coefficient[i][2][k] = b[k - NumofColinearity];
						}
					}

					for (int k = 0; k < nofcoeffs + 1; k++)
					{
						zeroposition[k] = 0;
					}
					break;
				default:
					break;
				}//end of switch
			}//end of j
		}//end of i

		 /***********************/
		 // Upper and Lower Bound
		 /***********************/
		tmpPMI.RndVarSeed = 6346538;
		//int abandon_count = 0;
		for (int TotalLoop = 0; TotalLoop < numofsamples; TotalLoop++)
		{
			cout << TotalLoop << endl;
			// start from Mar. Generate sample paths. loop.
			double sum = 0;

			simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample_1);

			int current_stage = 0;

			// generate basis functions based on the given sample path

			basisFunction(current_stage, stage, forwardCurveSample_1);

			stateSet[0] = 'O';

			lowerbound(tmpPMI.discFactor, forwardCurveSample_1, VFA_coefficient, tmpPMI, segmentStartingPoint);

			for (int i = 0; i < stage; i++) // i is from 0 to 1 to 2, because there is i+1 in the constraints.
			{
				greedy_policy_value[i] = rewardFunction(stateSet[i], action[i], i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0])*pow(discount_factor, i);
			}

			for (int Z = 0; Z < stage; Z++)
			{
				sum += greedy_policy_value[Z];
			}
			sample_Lower_Bound_Value[TotalLoop] = sum;
			lower_BD += sum;
		}// end of LOOP

		lower_BD = lower_BD / numofsamples;
		unbiased_upperbound = unbiased_upperbound / numofsamples;
		auto Lower_Bound_End_Time = high_resolution_clock::now();
		auto Total_End_Time = high_resolution_clock::now();
		//end_time_simulation_LB = clock();
		/********************************/
		//SEM for Upper Bound
		/********************************/
		for (int i = 0; i < numofsamples; i++)
		{
			diff_Upper_Bound += (sample_Upper_Bound_Value[i] - unbiased_upperbound)*(sample_Upper_Bound_Value[i] - unbiased_upperbound);
		}
		variance_UpperBound = diff_Upper_Bound / (numofsamples - 1);
		variance_UpperBound = sqrt(variance_UpperBound);
		variance_UpperBound = variance_UpperBound / sqrt(numofsamples);

		/********************************/
		//SEM for Lower Bound
		/********************************/
		for (int i = 0; i < numofsamples; i++)
		{
			diff_Lower_Bound += (sample_Lower_Bound_Value[i] - lower_BD)*(sample_Lower_Bound_Value[i] - lower_BD);
		}
		variance_LowerBound = diff_Lower_Bound / (numofsamples - 1);
		variance_LowerBound = sqrt(variance_LowerBound);
		variance_LowerBound = variance_LowerBound / sqrt(numofsamples);

		//*******************************************************************************************
		//Print
		//*******************************************************************************************
		//cout << "The estimated biased upper bound is " << model.get(GRB_DoubleAttr_ObjVal) << endl;
		cout << "The estimated unbiased upper bound is " << unbiased_upperbound << endl;
		cout << "The SEM for unbiased Upper Bound is " << variance_UpperBound << endl;
		cout << "The estimated lower bound is " << lower_BD << endl;
		cout << "The SEM for Lower Bound is " << variance_LowerBound << endl;
		//end_time = clock();
		auto Total_duration = duration_cast<seconds>(Total_End_Time - start_time);
		auto PCA_duration = duration_cast<seconds>(LP_start_time - start_time);
		auto PO_duration = duration_cast<seconds>(PO_stop_time - PO_start_time);
		auto LB_duration = duration_cast<seconds>(Lower_Bound_End_Time - Lower_Bound_Start_Time);
		//auto UB_duration = duration_cast<seconds>(Upper_Bound_End_Time - Upper_Bound_Start_Time);

		cout << "The running time is " << Total_duration.count() << endl;
		//cout << "The Number of Abandonment before the 5th stage is " << abandon_count << endl;

		//ofstream finalResult("C:\\Users\\boy1\\Desktop\\Result_PO_May.txt", ios::out);
		string finalResultPath = "./Final_Results_Greedy_PO_No_LSM_" + Month + ".txt";
		ofstream finalResult(finalResultPath, ios::out);
		finalResult << "The Unbiased UB is " << unbiased_upperbound << endl;
		finalResult << "The SEM for UB is " << variance_UpperBound << endl;
		finalResult << "The Unbiased LB is " << lower_BD << endl;
		finalResult << "The SEM for LB is " << variance_LowerBound << endl;
		finalResult << "Time for PCA: " << PCA_duration.count() << endl;
		finalResult << "Time for solving the PO-LP: " << PO_duration.count() << endl;
		finalResult << "Time for estimating the UB: " << UB_duration.count() << endl;
		finalResult << "Time for estimating the LB: " << LB_duration.count() << endl;
		finalResult << "Time for the whole process: " << Total_duration.count() << endl;
		//finalResult << "The Number of Abandonment before the 5th stage is " << abandon_count << endl;
		finalResult.close();
	}

	catch (GRBException e)
	{
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
		system("pause");
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
		if (state == 'O'&&action == 'P')
		{
			return (spotPrice_Ethanol - (corn_input_rate)*spotPrice_Corn - (NG_input_rate)*spotPrice_NG)*production + cost_for_production;
		}
		if (state == 'O'&&action == 'A')
			return salvage_value + cost_for_Abandonment;
		if (state == 'O'&&action == 'M')
			return cost_for_mothball;
		if (state == 'O'&&action == 'S')
			return cost_for_suspension;
		if (state == 'M1' && action == 'M')
			return cost_for_keepMothballing;
		if (state == 'M1'&& action == 'A')
			return salvage_value + cost_for_Abandonment;
		if (state == 'M1' && action == 'R')
			return cost_for_reactivation;
		if (state == 'R1' && action == 'R')
			return 0;
		if (state == 'R2' && action == 'P')
			return 0;
		if (state == 'NULL' && action == 'NULL')
			return 0;
		if (state == 'Empt' && action == 'Empt')
			return 0;
	}
	catch (GRBException e)
	{
		cout << "error" << endl;
		cout << state << " " << action << " " << current_stage << endl;
	}
}

void Initialization(vector<vector<vector<vector<double>>>>forwardCurveSample_0, vector<vector<vector<vector<double>>>> forwardCurveSample, double* discount_factor, int* tmpPMI_nStages, int*tmpPMI_nMarkets, int*tmpPMI_nCommodities)
{
	struct_PriceModelInfo <vector<vector<vector<double>>>> tmpPMI;

	ifstream priceModelCalibInputFile_0("/home/boy1/Ethanol_InputFile/priceModelCalibInputFile.txt", ios::in);
	string InitialForwardInputPath = "/home/boy1/Ethanol_InputFile/priceModelInitCondnInputFile_" + Month + ".txt";
	ifstream priceModelInitCondnInputFile_0(InitialForwardInputPath, ios::in);

	initializePriceModelStructures(priceModelCalibInputFile_0,
		priceModelInitCondnInputFile_0,
		forwardCurveSample_0,
		tmpPMI);

	*discount_factor = tmpPMI.discFactor;

	priceModelCalibInputFile_0.close();
	priceModelInitCondnInputFile_0.close();

	for (int totalloop = 0; totalloop < NumOfSamplePath; totalloop++)
	{
		forwardcurve_Total[totalloop] = new double***[tmpPMI.nStages + 1];
		for (int t = 0; t < tmpPMI.nStages + 1; t++) {
			forwardcurve_Total[totalloop][t] = new double**[tmpPMI.nCommodities];
			for (int c = 0; c < tmpPMI.nCommodities; c++) {
				forwardcurve_Total[totalloop][t][c] = new double*[tmpPMI.nMarkets];
				for (int m = 0; m < tmpPMI.nMarkets; m++) {
					forwardcurve_Total[totalloop][t][c][m] = new double[tmpPMI.nStages + 1];
					for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++)
					{
						forwardcurve_Total[totalloop][t][c][m][maturity] = 0; // forwardcurve stage + commodity + market + maturity
					}
				}
			}
		}
	}

	forwardCurveSample.resize(stage);
	for (int t = 0; t < stage; t++) {
		forwardCurveSample[t].resize(tmpPMI.nCommodities);
		for (int c = 0; c < tmpPMI.nCommodities; c++) {
			forwardCurveSample[t][c].resize(tmpPMI.nMarkets);
			for (int m = 0; m < tmpPMI.nMarkets; m++) {
				forwardCurveSample[t][c][m].resize(stage);
				for (int maturity = 0; maturity < stage; maturity++)
				{
					forwardCurveSample[t][c][m][maturity] = 0;
				}
			}
		}
	}

	tmpPMI.RndVarSeed = 5872345;
	cout << "Generating sample paths..." << endl;
	for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
	{
		simulateForwardCurveAntitheticFM(tmpPMI, forwardCurveSample);

		basisFunction(0, stage, forwardCurveSample);

		for (int current_stage = 1; current_stage < stage; current_stage++)
		{
			BasisSubtraction(current_stage, forwardCurveSample, tmpPMI, TotalLoop);
		}

		for (int t = 0; t < tmpPMI.nStages + 1; t++) {
			for (int c = 0; c < tmpPMI.nCommodities; c++) {
				for (int m = 0; m < tmpPMI.nMarkets; m++) {
					for (int maturity = 0; maturity < tmpPMI.nStages + 1; maturity++)
					{
						// half of the forward curve is 0.  
						forwardcurve_Total[TotalLoop][t][c][m][maturity] = forwardCurveSample[t][c][m][maturity]; // forwardcurve stage + commodity + market + maturity
					}
				}
			}
		}

		if (TotalLoop % 10000 == 0) {
			cout << "# of Samples: " << TotalLoop << endl;
		}
	}

	/*double sum = 0;
	for (int TotalLoop = 0; TotalLoop < NumOfSamplePath; TotalLoop++)
	{
	sum += forwardcurve_Total[TotalLoop][1][0][0][0];
	}

	cout << "Average is " << sum / NumOfSamplePath << endl;*/
	cout << "Performing PCA..." << endl;
	InputFunction();
}

void InputFunction()
{
	for (int i = 1; i < stage; i++)
	{
		vector<double> beta_var_value((nofcoeffs - 12 * (i - 1)) * NumOfSamplePath, 0);
		cout << "Stage: " << i << endl;
		int position = 0;
		for (int j = 0; j < NumOfSamplePath; j++)
		{
			for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
			{
				beta_var_value[position] = basisSubtraction[i][k * NumOfSamplePath + j];
				position++;
			}
		}

		//*******************************************************************
		//	PCA
		//*******************************************************************
		alglib::real_2d_array ptInput;
		ptInput.setcontent(NumOfSamplePath, nofcoeffs - 12 * (i - 1), &beta_var_value[0]);
		alglib::ae_int_t info;
		alglib::real_1d_array eigValues;
		alglib::real_2d_array eigVectors;

		alglib::pcabuildbasis(ptInput, NumOfSamplePath, nofcoeffs - 12 * (i - 1), info, eigValues, eigVectors);

		//double basisMatrix_test[nofcoeffs * nofcoeffs] = { 0 };
		double *results = new double[NumOfSamplePath * (nofcoeffs - 12 * (i - 1))];
		for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
		{
			for (int j = 0; j < nofcoeffs - 12 * (i - 1); j++)
			{
				EgenVectorMatrix[i][k * (nofcoeffs - 12 * (i - 1)) + j] = eigVectors[k][j];
			}
		}

		double alpha = 1;
		double beta = 0;
		char transA = 'T';
		char transB = 'T';

		int m = NumOfSamplePath;
		int n = nofcoeffs - 12 * (i - 1);//num of betas
		int k = n;//
		int LDA = k;
		int LDB = n;
		int LDC = m;
		int INCX = 1;
		int INCY = 1;

		cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, k, alpha, &beta_var_value[0], LDA, EgenVectorMatrix[i], LDB, beta, results, LDC);

		int *scale = new int[nofcoeffs - 12 * (i - 1)];
		int *scale_R = new int[nofcoeffs - 12 * (i - 1)];
		Order_Magnitude(results, i, scale, scale_R);

		position = 0;
		for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
		{
			for (int j = 0; j < NumOfSamplePath; j++)
			{
				basisSubtraction[i][position] = results[position] * powf(10, scale[k]);
				position++;
			}
		}

		for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
		{
			for (int j = 0; j < nofcoeffs - 12 * (i - 1); j++)
			{
				EgenVectorMatrix[i][k * (nofcoeffs - 12 * (i - 1)) + j] = eigVectors[k][j] * powf(10, scale[j]);
			}
		}

		copy(EgenVectorMatrix[i], EgenVectorMatrix[i] + (nofcoeffs - 12 * (i - 1))*(nofcoeffs - 12 * (i - 1)), Inv_EgenVectorMatrix[i]);

		delete[] scale;
		delete[] scale_R;
		delete[] results;

		/*string PATH = "./B_out_" + itos(i) + ".txt";
		ofstream B_output(PATH, ios::out);
		position = 0;
		for (int jj = 0; jj < nofcoeffs - 12 * (i - 1); jj++) {
		for (int ii = 0; ii < NumOfSamplePath; ii++) {
		B_output << setprecision(16) << basisSubtraction[i][position] << " ";
		position++;
		}
		B_output << endl;
		}*/
	}
}

void Order_Magnitude(double *results, int i, int *scale, int *scale_R)
{
	int position = 0;
	for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
	{
		double min_abs_value = powf(10, magnitude);
		for (int j = 0; j < NumOfSamplePath; j++)
		{
			if (fabs(results[position]) < min_abs_value)
			{
				min_abs_value = fabs(results[position]);
			}
			position++;
		}
		int scale_element = floorf(log10f(min_abs_value));
		if (scale_element <= magnitude)
		{
			scale[k] = magnitude - scale_element;
		}
	}
}

void Beta_Transformation(vector<vector<vector<double>>> &coefficient)
{
	for (int i = 1; i <= stage - 1; i++)
	{
		Beta_Update(i, coefficient[i]);
	}
}

void Beta_Update(int i, vector<vector<double>> &beta_var)
{
	double alpha = 1;
	double beta = 0;
	int m = nofcoeffs - 12 * (i - 1);
	int n = nofcoeffs - 12 * (i - 1);
	int LDA = m;
	int INCX = 1;
	int INCY = 1;

	vector<double> beta_var_Operational(beta_var[1]);
	vector<double> beta_var_Mothballed(beta_var[2]);

	vector<double> y(nofcoeffs - 12 * (i - 1), 0);
	vector<double> y_1(nofcoeffs - 12 * (i - 1), 0);

	/*string PATH = "./W_" + itos(i) + ".txt";
	ofstream B_output(PATH, ios::out);
	int position = 0;
	for (int jj = 0; jj < nofcoeffs - 12 * (i - 1); jj++) {
	for (int ii = 0; ii < nofcoeffs - 12 * (i - 1); ii++) {
	B_output << EgenVectorMatrix[i][position] << " ";
	position++;
	}
	B_output << endl;
	}
	B_output.close();

	ofstream Solution_output("./Beta_" + itos(i) + "_O"".txt", ios::out);
	for (int ii = 0; ii < m; ii++) {
	Solution_output << beta_var_Operational[ii + 1] << " ";
	}
	Solution_output.close();*/

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, EgenVectorMatrix[i], LDA, &beta_var_Operational[1], INCX, beta, &y[0], INCY);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, EgenVectorMatrix[i], LDA, &beta_var_Mothballed[1], INCX, beta, &y_1[0], INCY);

	for (int ii = 0; ii < m; ii++) {
		beta_var[1][ii + 1] = y[ii];
		beta_var[2][ii + 1] = y_1[ii];
	}
}

void inverse(double* A, int N)
{
	int *IPIV = new int[N + 1];
	//int LWORK = N*N;
	//double *WORK = new double[LWORK];
	//int INFO;

	//dgetrf_(&N, &N, A, &N, IPIV, &INFO);
	//dgetri_(&N, A, &N, IPIV, WORK, &LWORK, &INFO);
	LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, A, N, IPIV);
	LAPACKE_dgetri(LAPACK_COL_MAJOR, N, A, N, IPIV);

	delete[] IPIV;
	//delete WORK;
}

void Basis_Subtraction_Update(int i)
{
	double *SampleSubtraction = new double[nofcoeffs - 12 * (i - 1)]();
	//double *SampleSubtraction2 = new double[nofcoeffs - 12 * (i - 1)];
	int position = 0;
	for (int ii = 0; ii < nofcoeffs - 12 * (i - 1); ii++) {
		SampleSubtraction[ii] = basisSubtraction[i][ii*NumOfSamplePath];
	}
	//copy(basisSubtraction[i], basisSubtraction[i] + nofcoeffs - 12 * (i - 1), SampleSubtraction);

	double alpha = 1;
	double beta = 0;
	char transA = 'N';

	int m = nofcoeffs - 12 * (i - 1);
	int n = nofcoeffs - 12 * (i - 1);
	int LDA = m;
	int INCX = 1;
	int INCY = 1;

	double *y = new double[nofcoeffs - 12 * (i - 1)];

	for (int ii = 0; ii < nofcoeffs - 12 * (i - 1); ii++) {
		y[ii] = 0;
	}

	cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, EgenVectorMatrix[i], LDA, SampleSubtraction, INCX, beta, y, INCY);
	position = 0;

	for (int ii = 0; ii < nofcoeffs - 12 * (i - 1); ii++) {
		basisSubtraction[i][ii*NumOfSamplePath] = y[ii];
	}
	//copy(y, y + n, basisSubtraction[i]);

	delete[] y;
	delete[] SampleSubtraction;
}

void SampleTransformation()
{
	for (int i = 1; i <= stage - 1; i++)
	{
		Basis_Subtraction_Update(i);
	}
}

void BasisSubtraction(int current_stage, vector<vector<vector<vector<double>>>> &futures_price, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI, int TotalLoop)
{// input current_stage is i+1 from 1 to 5
	double s = 0;
	if (current_stage > stage - 1)
	{
		cout << "Current stage exceeds the limit !!!!!!!!!!!!!!" << endl;
	}
	else
	{
		int position = 0;
		for (int i = 0; i < 3; i++)
		{
			// F_i_j
			for (int j = 0; j < stage - current_stage; j++)
			{
				basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor * (basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1]);
				//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
				//b1[current_stage][0] += basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1];
				position++;
			}
		}

		// F_i_j*F_i_j
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < stage - current_stage; j++)
			{
				basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor*(basis2[current_stage][i][j] - basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1]);
				//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
				//b2[current_stage][0] += (basis2[current_stage][i][j] - basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1]);
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
					basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor*(basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
					//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
					//b3[current_stage][0] += (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
					position++;
				}
			}
		}

		// F_i_j_c* F_i_j*n
		for (int j = 0; j < stage - current_stage; j++)
		{
			basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor*(basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
			//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
			//b4[current_stage][0] += (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
			position++;
			basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor*(basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
			//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
			//b4[current_stage][0] += (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
			position++;
			basisSubtraction[current_stage][position * NumOfSamplePath + TotalLoop] = MI.discFactor*(basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
			//b4[current_stage][0] += (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
			//cout << basisSubtraction[current_stage][position][TotalLoop][0] <<" ";
			position++;
		}
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

void ReducedCost(vector<vector<vector<GRBConstr>>>& ConstraintPointer, vector<vector<double>>&Reduced_Cost, double**basisSubtraction) {
	for (int i = 1; i < stage; i++) {
		for (int j = 1; j < max_state; j++) {
			vector<double> dual_solution(ConstraintPointer[i][j].size(), 0);
			if (dual_solution.size() != 0) {
				//ofstream dualsolution("./dual_solution.txt", ios::out);
				for (int k = 0; k < ConstraintPointer[i][j].size(); k++) {
					dual_solution[k] = ConstraintPointer[i][j][k].get(GRB_DoubleAttr_Pi);
					//dualsolution << ConstraintPointer[i][j][k].get(GRB_DoubleAttr_Pi) << endl;
				}

				int NumOfActions = ConstraintPointer[i][j].size() / NumOfSamplePath;

				double alpha = 1;
				double two_alpha = 0.5;
				double negative_alpha = -1;
				double beta = 0;
				int m = NumOfSamplePath;
				int n = nofcoeffs - 12 * (i - 1);
				int LDA = m;
				//int LDA = n;
				int INCX = 1;
				int INCY = 1;

				if (NumOfActions == 1) {
					vector<double> temp_dual_solution(dual_solution);
					vector<double> y_1(dual_solution.size(), 0);
					vector<double> temp_y(dual_solution.size(), 0);
					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution[0], INCX, beta, &y_1[0], INCY);
					copy(y_1.begin(), y_1.end(), temp_y.begin());
					Reduced_Cost[i][j] = inner_product(y_1.begin(), y_1.end(), temp_y.begin(), 0.0);
				}
				else if (NumOfActions == 2) {
					vector<double> temp_dual_solution_1(dual_solution.begin(), dual_solution.begin() + NumOfSamplePath);
					vector<double> temp_dual_solution_2(dual_solution.begin() + NumOfSamplePath, dual_solution.end());

					vector<double> y_1(NumOfSamplePath, 0);
					vector<double> y_2(NumOfSamplePath, 0);

					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution_1[0], INCX, beta, &y_1[0], INCY);
					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution_2[0], INCX, beta, &y_2[0], INCY);

					transform(y_1.begin(), y_1.end(), y_2.begin(), y_1.begin(), plus<double>());

					copy(y_1.begin(), y_1.end(), y_2.begin());

					Reduced_Cost[i][j] = inner_product(y_1.begin(), y_1.end(), y_2.begin(), 0.0);
				}
				else if (NumOfActions == 3) {
					vector<double> temp_dual_solution_1(dual_solution.begin(), dual_solution.begin() + NumOfSamplePath);
					vector<double> temp_dual_solution_2(dual_solution.begin() + NumOfSamplePath, dual_solution.begin() + 2 * NumOfSamplePath);
					vector<double> temp_dual_solution_3(dual_solution.begin() + 2 * NumOfSamplePath, dual_solution.end());

					vector<double> y_1(NumOfSamplePath, 0);
					vector<double> y_2(NumOfSamplePath, 0);
					vector<double> y_3(NumOfSamplePath, 0);

					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution_1[0], INCX, beta, &y_1[0], INCY);
					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution_2[0], INCX, beta, &y_2[0], INCY);
					cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, &basisSubtraction[i][0], LDA, &temp_dual_solution_3[0], INCX, beta, &y_3[0], INCY);

					transform(y_1.begin(), y_1.end(), y_2.begin(), y_1.begin(), plus<double>());
					transform(y_1.begin(), y_1.end(), y_3.begin(), y_1.begin(), plus<double>());

					copy(y_1.begin(), y_1.end(), y_2.begin());

					Reduced_Cost[i][j] = inner_product(y_1.begin(), y_1.end(), y_2.begin(), 0.0);
				}
			}
		}
	}
}

GRBLinExpr penaltyFunction(int action, int state, int current_stage, int nofcoeffs, int TotalLoop)
{// input current_stage is i + time_lag_for_reactivation
	GRBLinExpr s = 0;
	if (current_stage > stage - 1) //shut down the penalty term
		return s;
	else
	{
		if ((state == 'O' && (action == 'P' || action == 'S')) || (state == 'M1' && action == 'R'))
		{
			int position = 0;
			// F_i_j
			for (int i = 0; i < 3; i++) // i from 0 to 2
			{
				for (int j = 0; j < stage - current_stage; j++) // current stage is 0; j is from 0 to 1
				{
					s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)// current stage is 0; j is from 0 to 1
				{
					s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
					position++;
				}
			}

			// F_i_j * F_i_j+1
			if (current_stage < stage - 1)  // current stage=0 and 1
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - 1 - current_stage; j++)
					{
						s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;

				s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;

				s += coeff_vars[current_stage][1][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;
			}
			return s;
		}

		if ((state == 'O' && action == 'M') || (state = 'M1' && action == 'M'))
		{
			int position = 0;
			// F_i_j
			for (int i = 0; i < 3; i++) // i from 0 to 2
			{
				for (int j = 0; j < stage - current_stage; j++) // current stage is 0; j is from 0 to 1
				{
					s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)// current stage is 0; j is from 0 to 1
				{
					s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
					position++;
				}
			}

			// F_i_j * F_i_j+1
			if (current_stage < stage - 1)
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < stage - current_stage - 1; j++)
					{
						s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;

				s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;

				s += coeff_vars[current_stage][2][position] * basisSubtraction[current_stage][position*NumOfSamplePath + TotalLoop];
				position++;
			}
			return s;
		}
	}
}

double PenaltyFunction(int action, int state, int current_stage, int nofcoeffs, vector<vector<vector<vector<double>>>> &futures_price, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI, int TotalLoop)
{// input current_stage is i+1 from 1 to 5
	double s = 0;
	if (current_stage > stage - 1)
		return s;
	else
	{
		if ((state == 'O' && (action == 'P' || action == 'S')) || (state == 'M1' && action == 'R'))
		{
			int position = 1; // position starting from 1 since the first coefficient is the constant 0;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//cout << coefficient[current_stage][1][position]<<" ";
					//b1[current_stage][0] += basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
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
						s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
						//cout << coefficient[current_stage][1][position]<<" ";
						//b3[current_stage][0] += (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][1][position]<<" ";
				//b4[current_stage][0] += (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][1][position]<<" ";
				//b4[current_stage][0] += (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//b4[current_stage][0] += (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
				//cout << coefficient[current_stage][1][position]<<" ";
				position++;
			}

			//cout << endl;
			return s;
		}


		if ((state == 'O' && action == 'M') || (state == 'M1' && action == 'M'))
		{
			int position = 1;
			// F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//	cout << coefficient[current_stage][2][position] << " ";
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//	cout << coefficient[current_stage][2][position] << " ";
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
						s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
						//	cout << coefficient[current_stage][2][position] << " ";
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;

				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;

				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;
			}
			//	cout << endl;
			return s;
		}
	}
}

double PenaltyFunction(int action, int state, int current_stage, int nofcoeffs, vector<vector<vector<double>>> &coefficient, int TotalLoop)
{// input current_stage is i+1 from 1 to 5
	double s = 0;
	if (current_stage > stage - 1)
		return s;
	else
	{
		if ((state == 'O' && (action == 'P' || action == 'S')) || (state == 'M1' && action == 'R'))
		{
			int position = 1; // position starting from 1 since the first coefficient is the constant 0;
			for (int i = 0; i < 3; i++)
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//cout << coefficient[current_stage][1][position]<<" ";
					//b1[current_stage][0] += basis1[current_stage][i][j] - basis1[current_stage - 1][i][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
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
						s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
						//cout << coefficient[current_stage][1][position]<<" ";
						//b3[current_stage][0] += (basis4[current_stage][i][j] - basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1]);
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][1][position]<<" ";
				//b4[current_stage][0] += (basis3[current_stage][j][0] - basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][1][position]<<" ";
				//b4[current_stage][0] += (basis3[current_stage][j][2] - basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1]);
				position++;

				s += coefficient[current_stage][1][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//b4[current_stage][0] += (basis3[current_stage][j][1] - basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1]);
				//cout << coefficient[current_stage][1][position]<<" ";
				position++;
			}

			//cout << endl;
			return s;
		}

		if ((state == 'O' && action == 'M') || (state == 'M1' && action == 'M'))
		{
			int position = 1;
			// F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//	cout << coefficient[current_stage][2][position] << " ";
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
					//	cout << coefficient[current_stage][2][position] << " ";
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
						s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
						//	cout << coefficient[current_stage][2][position] << " ";
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;

				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;

				s += coefficient[current_stage][2][position] * basisSubtraction[current_stage][(position - 1) * NumOfSamplePath + TotalLoop];
				//cout << coefficient[current_stage][2][position] << " ";
				position++;
			}
			//	cout << endl;
			return s;
		}
	}
}

//*****************************************************************************
// Dual bound
//*****************************************************************************
double upperbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI, double U_value[][max_state - 1])
{
	// 0 is operational
	// 1 is mothballed

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
					v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - PenaltyFunction('P', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, 0) + discount_factor * U_value[i + 1][0];
				}
				//Third action: mothball, only happens before stage 19. 
				if (i < stage - 1 - time_lag_reactivation)
				{
					v3 = rewardFunction('O', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - PenaltyFunction('M', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, 0) + discount_factor * U_value[i + 1][1];
				}

				// Forth action: Suspend
				if (i < stage - 1)
				{
					v4 = rewardFunction('O', 'S', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - PenaltyFunction('S', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, 0) + discount_factor * U_value[i + 1][0];
				}

				////////////////////////////////////////////////////
				//Decision
				////////////////////////////////////////////////////
				if (v1 >= v2 && v1 >= v3 && v1 >= v4) // the action is abandon // in the last stage, only abandon is available
				{
					U_value[i][0] = v1;
				}

				if (v2 >= v1 && v2 >= v3 && v2 >= v4) // the action is produce
				{
					U_value[i][0] = v2;
				}

				if (v3 >= v1 && v3 >= v2 && v3 >= v4) // the action is mothball
				{
					U_value[i][0] = v3;                     // jump to the M2 stage
				}

				if (v4 >= v1 && v4 >= v2 && v4 >= v3) // the action is suspend 
				{
					U_value[i][0] = v4;
				}

				break;
			case 'M1':
				if (i < stage - 1 - time_lag_reactivation) // reward for action keeping mothballing in state M2, only happen before 20.
				{
					v1 = rewardFunction('M1', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - PenaltyFunction('M', 'M1', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, 0) + discount_factor * U_value[i + 1][1];
				}
				v2 = rewardFunction('M1', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
				v3 = rewardFunction('M1', 'R', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) - PenaltyFunction('R', 'M1', i + time_lag_reactivation, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, 0) + discount_factor * U_value[i + time_lag_reactivation][0];

				if (v1 >= v2 && v1 >= v3) // action is M, keep mothballing
				{
					U_value[i][1] = v1;
				}

				if (v2 >= v1 && v2 >= v3) // action is abandon
				{
					U_value[i][1] = v2;
				}

				if (v3 >= v2 && v3 >= v1) // action is R, keep mothballing
				{
					U_value[i][1] = v3;
				}
				break;
			}//endl of switch
		} // end of j
	} // end of i
	return U_value[0][0];
}

//*****************************************************************************
//  Greedy Policy & Lower Bound
//*****************************************************************************
void lowerbound(double discount_factor, vector<vector<vector<vector<double>>>> &forwardCurveSample_1, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & tmpPMI, int segmentStartingPoint)
{
	int i = 0;// i is the stage. from 0 to 23.
	while (i < stage)
	{
		double v1 = -1000;// v1 is a small number.
		double v2 = -1000;
		double v3 = -1000;
		double v4 = -1000;

		switch (stateSet[i])// stateSet records the states for a greedy policy. from 0 to 23
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
			//if current state is "operation", we first calculate the total reward for 4 actions. And then we choose the action that can bring the largest reward. 

			//First action: abandon
			v1 = rewardFunction('O', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);

			//Second action: Produce
			if (i < stage - 1)
			{
				v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Approximation('P', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, segmentStartingPoint);
				//v2 = rewardFunction('O', 'P', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
			}
			//Third action: mothball, only happens before stage 19. 
			if (i < stage - 1 - time_lag_reactivation)
			{
				v3 = rewardFunction('O', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Approximation('M', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, segmentStartingPoint);
			}

			// Forth action: Suspend
			if (i < stage - 1)
			{
				v4 = rewardFunction('O', 'S', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Approximation('S', 'O', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, segmentStartingPoint);
			}
			if (v1 >= v2 && v1 >= v3 && v1 >= v4) // the action is abandon
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
				i++;                     // jump to the M2 stage
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
				v1 = rewardFunction('M1', 'M', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Approximation('M', 'M1', i + 1, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, segmentStartingPoint);
			}
			v2 = rewardFunction('M1', 'A', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]);
			v3 = rewardFunction('M1', 'R', i, forwardCurveSample_1[i][0][0][0], forwardCurveSample_1[i][1][0][0], forwardCurveSample_1[i][2][0][0]) + discount_factor * Approximation('R', 'M1', i + time_lag_reactivation, nofcoeffs, forwardCurveSample_1, coefficient, tmpPMI, segmentStartingPoint);

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

				/*action[i+1] = 'R';
				stateSet[i + 2] = 'R2';

				action[i+2] = 'P';
				stateSet[i + 3] = 'O';*/
				//i++;
			}
			break;
		}
		}
	}
}
/********************************************************************************/
// Approximation (Expected VFA)
/********************************************************************************/
double Approximation(int action, int state, int current_stage, int nofcoeffs, vector<vector<vector<vector<double>>>> &futures_price, vector<vector<vector<double>>> &coefficient, struct_PriceModelInfo<vector<vector<vector<double> > > > & MI, int segmentStartingPoint)
{ //input current_stage is i + 1 from 1 to 23
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
					s += coefficient[current_stage][1][position] * basis1[current_stage - 1][i][j + 1];
					position++;
				}
			}

			// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][1][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
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
						s += coefficient[current_stage][1][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][1][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1 + segmentStartingPoint];
				position++;
			}
			return s;
		}


		/*******************************************************************************************************************************************/

		if (state == 'O' && action == 'M')
		{
			int position = 0;
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
					s += coefficient[current_stage][2][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
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
						s += coefficient[current_stage][2][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1 + segmentStartingPoint];
				position++;
			}
			return s;
		}

		/*******************************************************************************************************************************************/
		if (state == 'M1' && action == 'M')
		{
			int position = 0;
			s += coefficient[current_stage][2][position];
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

			//// F_i_j*F_i_j
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < stage - current_stage; j++)
				{
					s += coefficient[current_stage][2][position] * basis2[current_stage - 1][i][j + 1] * loadingcoeffs1[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
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
						s += coefficient[current_stage][2][position] * basis4[current_stage - 1][i][j + 1] * loadingcoeffs3[current_stage - 1][i][0][j + 1 + segmentStartingPoint];
						position++;
					}
				}
			}

			// F_i_j_c* F_i_j*n
			for (int j = 0; j < stage - current_stage; j++)
			{
				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][0] * loadingcoeffs2[current_stage - 1][0][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][2] * loadingcoeffs2[current_stage - 1][2][0][j + 1 + segmentStartingPoint];
				position++;

				s += coefficient[current_stage][2][position] * basis3[current_stage - 1][j + 1][1] * loadingcoeffs2[current_stage - 1][1][0][j + 1 + segmentStartingPoint];
				position++;
			}
			return s;
		}
		/*******************************************************************************************************************************************/
		/********************************************************************************************************************************************/
		if (state == 'M1' && action == 'R')
		{
			int position = 0;
			s += coefficient[current_stage][1][position];
			position++;

			for (int i = 0; i < 3; i++) // i from 0 to 2
			{
				// F_i_j
				for (int j = 0; j < stage - current_stage; j++) // current stage is 0; j is from 0 to 1
				{
					s += coefficient[current_stage][1][position] * basis1[current_stage - time_lag_reactivation][i][j + time_lag_reactivation];
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

void generation_A_matrix(int loop_of_sample_path, double*** matrix_A)
{
	for (int current_stage = 1; current_stage < stage; current_stage++)
	{
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

void Generate_Beta_Matrix(double* results, double *Beta_Matrix, int i)
{
	long long position = 0;

	for (int ii = 1; ii < i; ii++) //NOTE :: use coefficients starting from stage 1. 
	{
		for (int j = 1; j < max_state; j++)
		{
			position += NumOfSamplePath * (nofcoeffs - 12 * (ii - 1));
		}
	}

	// For state O
	for (int k = 0; k < nofcoeffs - 12 * (i - 1); k++)
	{
		copy(results + k * NumOfSamplePath, results + k * NumOfSamplePath + NumOfSamplePath, Beta_Matrix + position + NumOfSamplePath * k);
	}
}
/**************************************************************************
* Simulates a forward curve sample for each stage using the factor model and antithetic variates
**************************************************************************/
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
		tmpMonth = (totStages - nTempStages) / MI.nDaysPerMonth;// from 0 to 22
		dayShift = (totStages - nTempStages) % MI.nDaysPerMonth;
		nHedgeStageIndex = (MI.nStartingMonth - 1 + tmpMonth) % 12; // from 0 to 11 

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
		for (int i = 0; i < nTempStages; i++)
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
						* exp(uHedgeMultiplier);
				}
			}
		}


		////////////////////////////////////////////////
		// generate the loading coeffs matrix///////////
		////////////////////////////////////////////////
		if (controlvalue == 0)
		{
			for (int i = 0; i < nTempStages; i++) //i is from 0 to 22
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
							= exp(uHedgeMultiplier);
					}
				}
			}
			//	//	///////////////////////////////////////
			//	//	//////////////////////////////////////
			//	//	//////////////////////////////////////
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
			if (nTempStages > 1)
			{
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
		}
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
	for (int c = 0; c < tmpPMI.loadingCoeffComm; c++)
	{
		tmpPMI.loadingCoeffs[c].resize(tmpPMI.loadingCoeffMkts);
		for (int mk = 0; mk < tmpPMI.loadingCoeffMkts; mk++)
		{
			tmpPMI.loadingCoeffs[c][mk].resize(tmpPMI.loadingCoeffMonths);
			for (int t = 0; t < tmpPMI.loadingCoeffMonths; t++)
			{
				tmpPMI.loadingCoeffs[c][mk][t].resize(tmpPMI.loadingCoeffMats*tmpPMI.loadingCoeffComm);
				for (int f = 0; f < tmpPMI.loadingCoeffMats*tmpPMI.loadingCoeffComm; f++)
				{
					tmpPMI.loadingCoeffs[c][mk][t][f].resize(
						tmpPMI.loadingCoeffMats);
					for (int mo = 0; mo < tmpPMI.loadingCoeffMats; mo++)
					{
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
	//tmpPMI.nStartingMonth = 1;

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

	//tmpPMI.nSimSamples = 12;
	//tmpPMI.nRegSamples = cmdLineParams.numRegSamples;

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