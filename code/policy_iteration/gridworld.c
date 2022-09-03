#include <stdio.h>
#include <math.h>

// The environment has 16 states, arranged as a 4x4 grid.
#define NSTATES 16
// There are four actions available in each state
typedef enum {LEFT,UP,RIGHT,DOWN} ACTION;
#define NACTIONS (DOWN+1-LEFT)
// The state transition probabilities are deterministic.
float P_sas[16][4][16] = {
  {
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 0, taking action LEFT
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 0, taking action UP
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 0, taking action RIGHT
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0} // starting in state 0, taking action DOWN
  },
  {
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 1, taking action LEFT
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 1, taking action UP
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 1, taking action RIGHT
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0} // starting in state 1, taking action DOWN
  },
  {
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 2, taking action LEFT
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 2, taking action UP
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 2, taking action RIGHT
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0} // starting in state 2, taking action DOWN
  },
  {
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 3, taking action LEFT
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 3, taking action UP
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 3, taking action RIGHT
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0} // starting in state 3, taking action DOWN
  },
  // second row
  {
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},// starting in state 4, taking action LEFT
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 4, taking action UP
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},// starting in state 4, taking action RIGHT
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0} // starting in state 4, taking action DOWN
  },
  {
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},// starting in state 5, taking action LEFT
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 5, taking action UP
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},// starting in state 5, taking action RIGHT
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0} // starting in state 5, taking action DOWN
  },
  {
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},// starting in state 6, taking action LEFT
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 6, taking action UP
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},// starting in state 6, taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0} // starting in state 6, taking action DOWN
  },
  {
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},// starting in state 7, taking action LEFT
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},// starting in state 7, taking action UP
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},// starting in state 7, taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0} // starting in state 7, taking action DOWN
  },
  // third row
  {
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},// starting in state 8, taking action LEFT
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},// starting in state 8, taking action UP
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},// starting in state 8, taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0} // starting in state 8, taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},// starting in state 9, taking action LEFT
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},// starting in state 9, taking action UP
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},// starting in state 9, taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0} // starting in state 9, taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},// starting in state 10 taking action LEFT
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},// starting in state 10 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},// starting in state 10 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0} // starting in state 10 taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},// starting in state 11 taking action LEFT
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},// starting in state 11 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},// starting in state 11 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1} // starting in state 11 taking action DOWN
  },
  // Fourth row
  {
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},// starting in state 12 taking action LEFT
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},// starting in state 12 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},// starting in state 12 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0} // starting in state 12 taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},// starting in state 13 taking action LEFT
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},// starting in state 13 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},// starting in state 13 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0} // starting in state 13 taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},// starting in state 14 taking action LEFT
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},// starting in state 14 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},// starting in state 14 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0} // starting in state 14 taking action DOWN
  },
  {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},// starting in state 15 taking action LEFT
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},// starting in state 15 taking action UP
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},// starting in state 15 taking action RIGHT
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1} // starting in state 15 taking action DOWN
  }
};

float p_sas(int s,int a, int sprime)
{
  return P_sas[s][a][sprime];
}

   
// reward function
float r(int s)
{
  if(s != 15)
    return -1;
  return 0;
}

// alternative
/* float r(int s) */
/* { */
/*   static float rewards[] = */
/*     { */
/*       -1,-1,-1,-1, */
/*       -1,-1,-1,-1, */
/*       -1,-1,-1,-1, */
/*       -1,-1,-1,0 */
/*     }; */
/*   return rewards[s]; */
/* } */

// The policy is a probability distribution over actions for each
// state. Initialize it to a random policy
float policy[16][4] =
  {
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS},
    {1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS,1.0/NACTIONS}
  };


float inner_bellman(int state, int action, float state_values[16])
{
  int i,j,k;
  float sa_value = 0.0;

  for(i=0;i<NSTATES;i++)
    sa_value += p_sas(state,action,i) * ( r(state) + 0.99 * state_values[i]);

  return sa_value;
}

// Evaluate the policy (1-step).  If the values have
// converged, return 1, else 0.
int evaluate_policy(float policy[16][4],float state_values[16],float delta)
{
  int i,j;
  int converged = 1;
  float new_values[16];
  // apply the Bellman equation to each state.
  for(i=0;i<NSTATES;i++)
    {
      // for each state, multiply the value of a state/action pair by
      // the probabily of choosing that action in the given state
      // under the given policy.
      new_values[i] = 0;
      for(j=0;j<NACTIONS;j++)
	new_values[i] += policy[i][j] * inner_bellman(i,j,state_values);
      // check for convergence
      if(fabs(new_values[i]-state_values[i]) > delta)
	converged = 0;
    }
  // copy new values into the state_values array
  for(i=0;i<NSTATES;i++)
    state_values[i] = new_values[i];
  return converged;
}

#define DETERMINISTIC

// Create a greedy policy based on the state values.
int improve_policy(float policy[16][4], float state_values[16])
{
  float new_policy[16][4] = {{0}};
  int i,j,k,action,count;
  int changed = 0;
  float max;
  // The value of taking action j in state i is the sum over each
  // possible resulting state times the probability of going to that
  // state.
  for(i=0;i<NSTATES;i++)
    for(j=0;j<NACTIONS;j++)
      for(k=0;k<NSTATES;k++)
	new_policy[i][j] += state_values[k] * p_sas(i,j,k);
  // For each state, choose the action with the higest value. There
  // could be ties.
  for(i=0;i<NSTATES;i++)
    {
      action = 0;
      for(j=1;j<NACTIONS;j++)
	if(new_policy[i][j] > new_policy[i][action])
	  action = j;
#ifdef DETERMINISTIC
	{      
	  for(j=0;j<NACTIONS;j++)
	    if(j==action)
	      new_policy[i][j] = 1.0;
	    else
	      new_policy[i][j] = 0.0;
	}
#else
	{
	  max = new_policy[i][action];
	  count = 0;
	  for(j=0;j<NACTIONS;j++)
	    if(new_policy[i][j] == max)
	      {
		new_policy[i][j] = 1.0;
		count++;
	      }
	    else
	      new_policy[i][j] = 0.0;
	  for(j=0;j<NACTIONS;j++)
	    new_policy[i][j] /= count;
	}
 #endif
    }
  
  // copy the new policy into the old policy
  for(i=0;i<NSTATES;i++)
    for(j=0;j<NACTIONS;j++)
      {
	if(policy[i][j] != new_policy[i][j])
	  changed = 1;
	policy[i][j] = new_policy[i][j];
      }
  return changed;
}


void print_values(float state_values[16])
{
  int i;
  printf("\n");
  for(i=0;i<NSTATES;i++)
    {
      printf(" %8.4f",state_values[i]);
      if(!((i+1)%4))
	printf("\n");
    }
}

void print_policy(float policy[16][4])
{
  int i,j;
  printf("\n             L    U    R    D\n");
  for(i=0;i<NSTATES;i++)
    {
      printf("State %02d:",i);
      for(j=0;j<NACTIONS;j++)
	printf(" %4.2f",policy[i][j]);
      printf("\n");
    }
}


int main()
{
  int i,policy_changed, converged;
  float state_values[16];

  print_policy(policy);
  
  // repeat evaluation/improvement until the policy is optimal.
  do
    {
      for(i=0;i<NSTATES;i++)
	state_values[i] = 0.0;
      
      // Evaluate the current policy, returning all of the state
      // values.  Repeat until the values converge.
      do
	{
	  print_values(state_values);
	  converged = evaluate_policy(policy, state_values, 0.001);
	} while (! converged);
      
      // Improve the policy by creating a new greedy policy based on
      // the state values.
      policy_changed =improve_policy(policy, state_values);
      print_policy(policy);
    }
  while(policy_changed);


  
  
  return 0;
}
