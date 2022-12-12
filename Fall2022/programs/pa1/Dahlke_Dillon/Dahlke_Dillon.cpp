/***************************************************************************
*	CSC 449 - Advanced Topics Artificial Intelligence
* 
*	Programming Assignment 1 - Gridworld Problem
* 
*	Author - Dillon Dahlke
*
*	This program implements a modified "Gridworld" problem to find the 
*	optimal policy and value function. This is achieved by using the 
*	Bellman equation. This program makes use of the 
*	deterministic element of the Gridworld to simplify the Bellman equation.
*
*
*
*
*
*****************************************************************************/
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>

using namespace std;

#define NUMSTATES 16
#define NUMACTIONS 4
#define GAMMA 0.95
#define THETA 0.001
#define P = 0.25


enum action {U,D,L,R};

//Because the environment is deterministic, this function defines movement
int movement(int s, action a)
{
	if (s == 8 || s == 15)
		return 15;
	else if (a == U)
	{
		if (s > 3)
			return s - 4;
		else
			return s;
	}
	else if (a == D)
	{
		if (s < 12)
			return s + 4;
		else
			return s;
	}
	else if (a == L)
	{
		if (s % 4 != 0)
			return s - 1;
		else
			return s;
	}
	else if (a == R)
	{
		if ((s + 1) % 4 != 0)
			return s + 1;
		else
			return s;
	}
}

//The function that determines the reward from taking a certain action
float reward(int s, action a)
{
	if (s == 8)
	{
		return -2;
	}
	else if (s == 15 && (a == D || a == R))
	{
		return 0;
	}
	else
		return -1;
}
//This function implements the basic Bellman algorithm
float bellman(float v[], int s)
{
	int i = 1;
	float temp = 0;
	int sprime = movement(s,action(0));
	float max = reward(s,action(0))+ GAMMA*v[sprime];
	for (i = 1; i < NUMACTIONS; i++)
	{
		sprime = movement(s, action(i));
		temp = reward(s, action(i)) + GAMMA * v[sprime];

		if (temp > max)
			max = temp;
	}
	return max;
}

//This function prints out the state values of the system
void print_values(float v[])
{
	int i = 0;

	for (i = 0; i < NUMSTATES; i++)
	{
		if (i % 4 == 0)
			cout << endl;
		cout << setw(10) << v[i];

	}
}

int main()
{
	float v[NUMSTATES];
	float vnew[NUMSTATES];
	
	int i = 0;
	int j = 0;
	float delta = 0;
	float temp = 0;

	//initialize the value function to all zero
	for (i = 0; i < NUMSTATES; i++)
		v[i] = 0;

	//Update the function until it converges
	do
	{
		delta = 0;
		print_values(v);
		cout << endl << endl;
		//Update all the values of states
		for (j = 0; j < NUMSTATES; j++)
		{
			vnew[j] = bellman(v, j);
		}
		for (j = 0; j < NUMSTATES; j++)
		{
			temp = abs(v[j] - vnew[j]);
			if (temp > delta)
				delta = temp;
		}
		//Copy the new array over
		for (j = 0; j < NUMSTATES; j++)
			v[j] = vnew[j];

	} while (delta > THETA);

	//Print formatting for the value function
	cout << endl << endl << "Value Function" << endl;
	cout << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-" << endl;
	print_values(v);
	cout << endl << endl << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-" << endl;

	//Find the optimal policy
	for (i = 0; i < NUMSTATES; i++)
	{
		float temp = 0;
		int sprime = movement(i, action(0));
		float max = v[sprime];
		cout << "Actions for state " << i << ": ";
		for (j = 1; j < NUMACTIONS; j++)
		{
			sprime = movement(i, action(j));
			temp = v[sprime];
			if (temp > max)
				max = temp;
		}
		for (j = 0; j < NUMACTIONS; j++)
		{
			sprime = movement(i, action(j));
			temp = v[sprime];
			if (temp == max)
			{
				switch (j)
				{
				case 0:
					cout << "UP ";
				case 1:
					cout << "DOWN ";
				case 2:
					cout << "LEFT ";
				case 3:
					cout << "RIGHT ";
				}
			}
		}


		cout << endl;


	}
	
	return 0;
}
