#include <iostream>
#include <iomanip>

using namespace std;

//discount factor
float GAMMA = 0.95f;

/******************************************************************************
*                         Function Prototypes
******************************************************************************/
void fillActions(int steps[][4][16]);
void fillPolicy(float policy[][4]);
void fillRewards(int rewards[][4]);
void printPolicy(float policy[][4]);
void printPolicyCute(float policy[][4]);
void printReward(int rewards[][4]);
void printStateValues(float state_values[]);
bool evalPolicy(float policy[][4], float state_values[], float theta, 
    int steps[][4][16], int rewards[][4]);
bool improvePolicy(float policy[][4], float state_values[], int steps[][4][16], 
    int rewards[][4]);
int getReward(int state, int action, int rewards[][4]);
int nextState(int state, int action, int steps[][4][16]);


/** ***************************************************************************
 * @author Dakota Walker
 *****************************************************************************/
int main()
{
    int n = 16;  //amount of states
    int i = 0;
    bool converged = true;
    bool policy_changed = false;
    int rewards[16][4] = { 0 };     //r(s,a)
    int steps[16][4][16] = { 0 };   //S'(s,a)
    float policy[16][4] = { 0 };    //Policy probability
    float state_value[16] = { 0 };  //state values

    //fill in default policy, s,a,s' table, and r(s,a) table.
    fillPolicy(policy);
    fillActions(steps);
    fillRewards(rewards);

    //repeat policy evaluation/improvement until optimal
    do
    {
        //set state values = 0
        for (i = 0; i < n; i++)
        {
            state_value[i] = 0.0;
        }

        //loop through states until converged
        do
        {
            printStateValues(state_value);
            converged = evalPolicy(policy, state_value, 0.0001f, steps, rewards);
        } while (!converged);

        //improve the policy, returns false if can't improve
        policy_changed = improvePolicy(policy, state_value, steps, rewards);

        printPolicy(policy);

    } while (!policy_changed);

    printPolicyCute(policy);

    return 0;
}


void fillPolicy(float policy[][4])
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            policy[i][j] = 0.25;
        }
    }
}


void fillActions(int steps[][4][16])
{
    //state 0
    steps[0][0][0] = 1;   // LEFT
    steps[0][1][0] = 1;   // UP
    steps[0][2][1] = 1;   // RIGHT
    steps[0][3][4] = 1;   // DOWN

    //state 1
    steps[1][0][0] = 1;   // LEFT
    steps[1][1][1] = 1;   // UP
    steps[1][2][2] = 1;   // RIGHT
    steps[1][3][5] = 1;   // DOWN

    //state 2
    steps[2][0][1] = 1;   // LEFT
    steps[2][1][2] = 1;   // UP
    steps[2][2][3] = 1;   // RIGHT
    steps[2][3][6] = 1;   // DOWN

    //state 3
    steps[3][0][2] = 1;   // LEFT
    steps[3][1][3] = 1;   // UP
    steps[3][2][3] = 1;   // RIGHT
    steps[3][3][7] = 1;   // DOWN

    //state 4
    steps[4][0][4] = 1;   // LEFT
    steps[4][1][0] = 1;   // UP
    steps[4][2][5] = 1;   // RIGHT
    steps[4][3][8] = 1;   // DOWN

    //state 5
    steps[5][0][4] = 1;   // LEFT
    steps[5][1][1] = 1;   // UP
    steps[5][2][6] = 1;   // RIGHT
    steps[5][3][9] = 1;   // DOWN

    //state 6
    steps[6][0][5] = 1;   // LEFT
    steps[6][1][2] = 1;   // UP
    steps[6][2][7] = 1;   // RIGHT
    steps[6][3][10] = 1;   // DOWN

    //state 7
    steps[7][0][6] = 1;   // LEFT
    steps[7][1][3] = 1;   // UP
    steps[7][2][7] = 1;   // RIGHT
    steps[7][3][11] = 1;   // DOWN

    //state 8
    steps[8][0][15] = 1;   // SPECIAL LEFT
    steps[8][1][4] = 1;   // UP
    steps[8][2][9] = 1;   // RIGHT
    steps[8][3][12] = 1;   // DOWN

    //state 9
    steps[9][0][8] = 1;   // LEFT
    steps[9][1][5] = 1;   // UP
    steps[9][2][10] = 1;   // RIGHT
    steps[9][3][13] = 1;   // DOWN

    //state 10
    steps[10][0][9] = 1;   // LEFT
    steps[10][1][6] = 1;   // UP
    steps[10][2][11] = 1;   // RIGHT
    steps[10][3][14] = 1;   // DOWN

    //state 11
    steps[11][0][10] = 1;   // LEFT
    steps[11][1][7] = 1;   // UP
    steps[11][2][11] = 1;   // RIGHT
    steps[11][3][15] = 1;   // DOWN

    //state 12
    steps[12][0][12] = 1;   // LEFT
    steps[12][1][8] = 1;   // UP
    steps[12][2][13] = 1;   // RIGHT
    steps[12][3][12] = 1;   // DOWN

    //state 13
    steps[13][0][12] = 1;   // LEFT
    steps[13][1][9] = 1;   // UP
    steps[13][2][14] = 1;   // RIGHT
    steps[13][3][13] = 1;   // DOWN

    //state 14
    steps[14][0][13] = 1;   // LEFT
    steps[14][1][10] = 1;   // UP
    steps[14][2][15] = 1;   // RIGHT
    steps[14][3][14] = 1;   // DOWN

    //state 15
    steps[15][0][14] = 1;   // LEFT
    steps[15][1][11] = 1;   // UP
    steps[15][2][15] = 1;   // RIGHT
    steps[15][3][15] = 1;   // DOWN

    return;
}


void fillRewards(int rewards[][4])
{
    int i, j = 0;
    for (i = 0; i < 16; i++)
    {
        for (j = 0; j < 4; j++)
        {
            rewards[i][j] = -1;
        }
    }

    //set special state/action rewards
    rewards[8][0] = -2;
    rewards[15][2] = 0;
    rewards[15][3] = 0;


    return;
}


bool evalPolicy(float policy[][4], float state_values[], float change, 
    int steps[][4][16], int rewards[][4])
{
    int s, a, r, s_prime = 0;
    float old_v;
    float biggestChange = 0;
    float new_v[16] = { 0 };

    //for each state
    for (s = 0; s < 16; s++)
    {
        //save old value
        old_v = state_values[s];

        //for all eligible actions
        for (a = 0; a < 4; a++)
        {
            //get next state based on action
            s_prime = nextState(s, a, steps);
            r = getReward(s, a, rewards);

            //sum up new values
            new_v[s] += policy[s][a] * (r + (GAMMA * state_values[s_prime]));
        }
        biggestChange = max(biggestChange, abs(old_v - new_v[s]));
    }

    for (s = 0; s < 16; s++)
    {
        state_values[s] = new_v[s];
    }

    //check if biggest change less than the acceptable change for convergance
    if (biggestChange <= change)
    {
        return true;
    }
    return false;
}


bool improvePolicy(float policy[][4], float state_values[], int steps[][4][16],
    int rewards[][4])
{
    int s, a, r, s_prime, i, count = 0;
    float best = -99999;
    float oldPolicy[16][4];
    float v[4];
    bool policy_stable = true;

    //for every state
    for (s = 0; s < 16; s++)
    {
        //loop through actions
        for (a = 0; a < 4; a++)
        {
            //save old policy
            oldPolicy[s][a] = policy[s][a];

            //get next state based on action
            s_prime = nextState(s, a, steps);
            r = getReward(s, a, rewards);

            v[a] = r + (GAMMA * state_values[s_prime]);

            //save best value and action
            if (v[a] == best)
            {
                count++;
                best = v[a];
            }
            else if (v[a] > best)
            {
                count = 1;
                best = v[a];
            }
        }

        //TODO: Break off into own function.
        //If a new best action was found, set the probability of that
        // action to 1 and other 3 to zero. Deterministic.
        if (count > 0)
        {
            for (i = 0; i < 4; i++)
            {
                if (v[i] == best)
                {
                    policy[s][i] = (float)1.0 / (float)count;
                }
                else
                {
                    policy[s][i] = 0;
                }

                if (oldPolicy[s][i] != policy[s][i])
                {
                    policy_stable = false;
                }
            }
            best = -99999;
            count = 0;
        }
    }

    return policy_stable;
}


int nextState(int state, int action, int steps[][4][16])
{
    for (int i = 0; i < 16; i++)
    {
        if (steps[state][action][i] != 0)
        {
            return i;
        }
    }

    return state;
}


int getReward(int state, int action, int rewards[][4])
{
    return rewards[state][action];
}


void printPolicy(float policy[][4])
{
    int i, j, count = 0;

    //header
    cout << endl << endl << setw(12) << left << "Policy: " << setw(6) << "L" 
        << setw(6) << "U" << setw(6) << "R" << setw(6) << "D" << endl;

    //for every state
    for (i = 0; i < 16; i++)
    {
        cout << setw(6) << left << "State " << setw(2) << right << i << ": ";
        //for every action
        for (j = 0; j < 4; j++)
        {
            cout << setw(6) << fixed << setprecision(2) << policy[i][j];
            count++;

            if (count == 4)
            {
                cout << endl;
                count = 0;
            }
        }
    }
    cout << endl;

    return;
}


void printPolicyCute(float policy[][4])
{
    int i, j, count = 0, temp = 0;
    float max = 0;
    char actions[4] = { '<', '^', '>', 'v' };

    cout << endl << endl << "An Optimal Policy: " << endl << '\t';
    //for each state
    for (i = 0; i < 16; i++)
    {
        //for each action
        for (j = 0; j < 4; j++)
        {
            //save best action
            if (policy[i][j] > max)
            {
                max = policy[i][j];
                temp = j;
            }
        }
        cout << actions[temp] << " ";
        max = 0.0;
        count++;

        if (count == 4)
        {
            cout << endl << '\t';
            count = 0;
        }
    }
    cout << endl;

    return;
}


void printStateValues(float state_values[])
{
    int i = 0;
    int count = 0;

    //print state_values
    for (i = 0; i < 16; i++)
    {
        cout << setw(7) << fixed << setprecision(3) << state_values[i];
        count++;

        if (count == 4)
        {
            cout << endl;
            count = 0;
        }

    }

    cout << endl << endl;

    return;
}


/*** DEBUG FUNCTION ***/
void printReward(int rewards[][4])
{
    int i, j = 0;
    char actions[4] = { 'L', 'U', 'R', 'D' };

    cout << endl << "Reward for actions in states: " << endl;

    for (i = 0; i < 16; i++)
    {
        cout << "State" << i;

        for (j = 0; j < 4; j++)
        {
            cout << " " << actions[j] << "=" << setw(2) << rewards[i][j];
        }
        cout << endl;
    }

    return;
}


// DEBUG CODE snippets
/*
    //steps[8][0][8] = 1;   // NORMAL LEFT movement for state 8

    //Useful reward control case where state 0 and 15 act like terminating points
    //state 0
    arr[0][0] = 0;
    arr[0][1] = 0;
    arr[1][0] = 0;
    arr[4][1] = 0;
    //state 15
    arr[15][2] = 0;
    arr[15][3] = 0;
    arr[14][2] = 0;
    arr[11][3] = 0;


    char actions[4] = { 'L', 'U', 'R', 'D' };
    cout << "Current State: " << s << endl
        << "Action: " << actions[a] << endl
        << "Next State: " << s_prime << endl
        << "Reward: " << r << endl
        << "Value: " << v << endl << endl;
*/