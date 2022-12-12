#include <iostream>
#include <iomanip>
#include <bits/stdc++.h>

using namespace std;

enum action {UP, DOWN, LEFT, RIGHT};

int bellmanEquation(int s, action a, int sPrime, int r);
int policyChance(int state, action a, int policy);
double updatePolicy(int curState, action a, double *valState);
int Move(int curState, action a);
int getReward(int state, action a);
double getStateValue(double *valState, int state, double gamma, double policySpread[4]);
void printStateValue(double* valState);
void printPolicy(double policy[16][4]);
void printDeterministicPolicy(double policy[16][4]);
void printStochasticPolicy(double policy[16][4]);
int prog(double gamma, double accuracyThreshold);
int calculateDeterministicPolicies(double policy[16][4]);

int main ()
{
	double gamma = 0.95; //Value given in assignment
	double accuracyThreshold = 0.001;
	int totalDeterministic = 0;
	totalDeterministic = prog(gamma, accuracyThreshold);
	//totalDeterministic = prog(0.999, 0.0001);

	cout << "Number of Optimal Deterministic Policies for gamma = 0.95: ";
	cout << totalDeterministic << endl;
	cout << "Calculated Number of Optimal Deterministic Policies for gamma = 1.0: 1296" << endl;
	return 0;
}
int prog(double gamma, double accuracyThreshold)
{
	//declare variables
	double delta = accuracyThreshold;
	int numStates = 16;
	bool isPolicyStable = false;
	int policyCounter = 0;
	//Arrays
	double valState [16] = {};
	double tempValState [16] = {};
	//Policies (2D array of [state][action])
	double policy[16][4];
	double tempPolicy[16][4];

	//initialize Policies
	for (int i = 0; i < numStates; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			policy[i][j] = 0.25;
			tempPolicy[i][j] = 0.25;
		}
	}
	
	//Policy Evaluation
	//Loop until (delta < accThresh)
	while (isPolicyStable == false)
	{
		cout << "Counter: " << ++policyCounter << endl;
		delta = accuracyThreshold;
		while (delta >= accuracyThreshold)
		{
			delta = 0;
			//Loop over states
			for (int i = 0; i < numStates; i++)
			{
				//cout << i << ", ";
				//v <- V(s)
				tempValState[i] = valState[i];
				//V(s) <- bellman equation(pt2)
				valState[i] = getStateValue(tempValState, i, gamma, policy[i]);
				//delta <- max(delta, |v-V(s)|)
				if (delta < (abs(tempValState[i] - valState[i])))
				{
					//cout << "TempVal: " << tempValState[i] << endl;
					//cout << "Val:" << valState [i] << endl;;
					delta = abs(tempValState[i] - valState[i]);
					//delta += 1;
				}

			}
			//cout << "Delta: ";
			//cout << setprecision(5) << fixed;
			//cout << setw(5) << delta << endl;
		}
		printStateValue(valState);
		//Policy Improvement
	
		isPolicyStable = true;
		//for each state
		for (int i = 0; i < numStates; i++)
		{
			//old action <- Pi(s)
			tempPolicy[i][0] = policy[i][0];
			tempPolicy[i][1] = policy[i][1];
			tempPolicy[i][2] = policy[i][2];
			tempPolicy[i][3] = policy[i][3];
			//pi(s) <- argmax_a(bellman equation_pt2)
			//cout << "Before" << endl;
			//printPolicy(policy);
			policy[i][UP] = updatePolicy(i, UP, valState);
			policy[i][DOWN] = updatePolicy(i, DOWN, valState);
			policy[i][LEFT] = updatePolicy(i, LEFT, valState);
			policy[i][RIGHT] = updatePolicy(i, RIGHT, valState);
			//cout << "After" << endl;
			//printPolicy(policy);
			for (int j = 0; j < 4; j++)
			{
				//if (old-action != pi(s))
				//if (abs(tempPolicy[i][j] - policy[i][j]) < accuracyThreshold)
				if (tempPolicy[i][j] != policy[i][j])
				{
					isPolicyStable = false;
				}
			}
		}
		if (isPolicyStable)
			cout << "Final Policy" << endl;
		printPolicy(policy);
	}
	printStochasticPolicy(policy);
	printDeterministicPolicy(policy);
	return calculateDeterministicPolicies(policy);
}


int bellmanEquation(int s, action a, int sPrime, int r)
{
	return 0;
}
int policyChance(int state, action a, int policy)
{
	return 0;
}
double updatePolicy(int curState, action a, double *valState)
{
	int numActions = 1;
	double actions[4];
	double threshold = 0.00001;
	//for every action
	for (int i = 0; i < 4; i++)
	{
		//fill array with surrounding values
		actions[i] = valState[Move(curState, action(i))]; 
	}
	sort(actions, actions + (sizeof(actions) / sizeof(actions[0])));
	if (abs(actions [2] - actions[3]) < threshold)
	{
		numActions++;
		if (abs(actions [1] - actions[3]) < threshold)
		{
			numActions++;
			if (abs(actions [0] - actions[3]) < threshold)
			{
				numActions++;
			}
		}
	}
	if (valState[Move(curState, a)] == actions[3])
	{
		//cout << "State: " << curState << ", ";
		//cout << "Num Actions: " << numActions << ", ";
		//cout << "Direction: " << a << ", " << endl;
		return 1.0/numActions;
	}
	return 0;
}
int Move(int curState, action a)
{
	int moveArr[16][4] = { //UP, DOWN, LEFT, RIGHT
	 0,  4,  0,  1, 
	 1,  5,  0,  2,
	 2,  6,  1,  3,
	 3,  7,  2,  3,
	 0,  8,  4,  5,
	 1,  9,  4,  6,
	 2, 10,  5,  7,
	 3, 11,  6,  7,
	 4, 12, 15,  9,
	 5, 13,  8, 10,
	 6, 14,  9, 11,
	 7, 15, 10, 11,
	 8, 12, 12, 13,
	 9, 13, 12, 14,
	10, 14, 13, 15,
	11, 15, 14, 15
	};
	return moveArr[curState][a];
}
int getReward(int state, action a)
{
	if (state == 8 && a == LEFT)
		return -2;
	if (state == 15 && ((a == RIGHT) || (a == DOWN)))
	       return 0;	
	return -1;
}
double getStateValue(double *valState, int state, double gamma, double policySpread[4])
{
	double total = 0;
	int newState = 0;
	double reward = 0;
	//for every action
	for (int i = 0; i < 4; i++)
	{
		if (policySpread[i] != 0)
		{
			//get new state
			newState = Move(state, action(i));
			//get reward
			reward = getReward(state, action(i));
			//sum equations (chance of getting to state * reward + discount reward);
			total += policySpread[i] * (reward + gamma * valState[newState]);
		}
	}
	//total += valState[state];
	return total;
}
void printStateValue(double valState[16])
{
	cout << setprecision(4) << fixed;
	for (int i = 0; i < 16; i++)
	{
		cout << '[' << (valState[i] - valState[15]) << ']';
		if (i % 4 == 3)
			cout << endl;
	}
	cout << "------------------------------------" << endl;
}
void printPolicy(double policy[16][4])
{
	for (int i = 0; i < 16; i++)
	{
		cout << '[';
		if (policy[i][UP] > 0)
			cout << '^';
		else
			cout << ' ';
		if (policy[i][DOWN] > 0)
			cout << 'v';
		else
			cout << ' ';
		if (policy[i][LEFT] > 0)
			cout << '<';
		else
			cout << ' ';
		if (policy[i][RIGHT] > 0)
			cout << '>';
		else
			cout << ' ';
		cout << ']';
		if (i % 4 == 3)
			cout << endl;
	}
	cout << "------------------------------------" << endl;
}
void printDeterministicPolicy(double policy[16][4])
{
	cout << "Deterministic Policy:" << endl;
	for (int i = 0; i < 16; i++)
	{
		cout << '[';
		if (policy[i][UP] > 0)
			cout << '^';
		else if (policy[i][DOWN] > 0)
			cout << 'v';
		else if (policy[i][LEFT] > 0)
			cout << '<';
		else if (policy[i][RIGHT] > 0)
			cout << '>';
		else
			cout << ' ';
		cout << ']';
		if (i % 4 == 3)
			cout << endl;
	}
	cout << "------------------------------------" << endl;
}
void printStochasticPolicy(double policy[16][4])
{
	cout << "Stochastic Policy:" << endl;
	for (int i = 0; i < 16; i++)
	{
		cout << setprecision(2);
		cout << '[';
		if (policy[i][UP] > 0)
			cout << policy[i][UP] << '^';
		else
			cout << "     ";
		if (policy[i][DOWN] > 0)
			cout << policy[i][DOWN] << 'v';
		else
			cout << "     ";
		if (policy[i][LEFT] > 0)
			cout << policy[i][LEFT] << '<';
		else
			cout << "     ";
		if (policy[i][RIGHT] > 0)
			cout << policy[i][RIGHT] << '>';
		else
			cout << "     ";
		cout << ']';
		if (i % 4 == 3)
			cout << endl;
	}
	cout << "------------------------------------" << endl;
}
int calculateDeterministicPolicies(double policy[16][4])
{
	int total = 1;
	for (int i = 0; i < 16; i++)
	{
		int pathCount = 0;
		for (int j = 0; j < 4; j++)
		{
			if (policy[i][j] > 0)
			{
				pathCount++;
			}
		}
		total *= pathCount;

	}
	return total;
}
