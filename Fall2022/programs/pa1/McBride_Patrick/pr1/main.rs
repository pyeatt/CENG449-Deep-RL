struct StateRewardPair {x: usize, y: usize, reward: f64}
struct PolicyStabilityPair {pi: [[[f64;4];4];4], stable: bool}

// Takes - State values
// Prints - Grid of state values.
fn print_state_values(state_values: [[f64;4];4]){
    for i in 0..4{
        for j in 0..4{
            print!("{:.2}   ", state_values[i][j])
        }
        println!("\n");
    }
}

// Takes - A policy
// Prints - Deterministic policy.
fn print_deterministic_policy(policy: [[[f64;4];4];4]){
    let (mut max, mut action);
    for i in 0..4{
        for j in 0..4{
            max = policy[0][i][j];
            action = 0;
            for a in 0..4{
                if policy[a][i][j] > max{
                    max = policy[a][i][j];
                    action = a;
                }
            }
            if action == 0{print!("Left     ");}
            else if action == 1{print!("Up       ");}
            else if action == 2{print!("Right    ");}
            else {print!("Down     ");}
        }
        println!("\n");
    }
}

// Takes - A policy
// Prints - Stochastic policy.
fn print_stochastic_policy(policy: [[[f64;4];4];4]){

    println!("Probability Action: LEFT is taken");
    for i in 0..4{
        for j in 0..4{
            print!("{:.2}   ", policy[0][i][j])
        }
        println!();
    }

    println!("Probability Action: UP is taken");
    for i in 0..4{
        for j in 0..4{
            print!("{:.2}   ", policy[1][i][j])
        }
        println!();
    }

    println!("Probability Action: RIGHT is taken");
    for i in 0..4{
        for j in 0..4{
            print!("{:.2}   ", policy[2][i][j])
        }
        println!();
    }

    println!("Probability Action: DOWN is taken");
    for i in 0..4{
        for j in 0..4{
            print!("{:.2}   ", policy[3][i][j])
        }
        println!();
    }
}

// Takes - A policy and values
// Prints - Formatted results
fn print_formatted_results( d_policy: [[[f64;4];4];4], d_values: [[f64;4];4],s_policy: [[[f64;4];4];4], s_values: [[f64;4];4]){
    println!("###############UNDERGRAD################");
    println!("Here is one optimal deterministic policy");
    print_deterministic_policy(d_policy);
    println!("Here are the state values for \nthat for that policy.");
    print_state_values(d_values);
    println!("########################################\n\n");
    println!("##################GRAD##################");
    print!("The number of optimal deterministic policies\nfor this problem: ");
    println!("{:?}\n",calculate_number_deterministic_policies(d_values));
    println!("Here is one optimal stochastic policy \nfor this problem.");
    print_stochastic_policy(s_policy);
    println!();
    println!("Here are the values for the stochastic policy.");
    print_state_values(s_values);
    println!();
    println!("########################################");
}

// Takes - State and action.
// Returns - The next state and reward for transitioning to that state.
fn get_next_state(xs: usize, ys: usize, action: usize)-> StateRewardPair {
    let mut next_state = StateRewardPair { x: xs, y: ys, reward: (-1.0)};

    // Left
    if action == 0 {
        // State 8 - Teleport to the terminal state
        if (xs == 0) && (ys == 2){
            next_state.x = 3;
            next_state.y = 3;
            next_state.reward = -2.0;
        }
        else if xs > 0 {
            next_state.x -= 1;
        }
    }

    // Up
    else if action == 1 {
        if ys > 0 {
            next_state.y -= 1;
        }
    }

    // Right
    else if action == 2 {
        // State 15 - Trying to move out of bounds
        if (xs == 3) && (ys == 3){
            next_state.reward = 0.0;
        }
        else if xs < 3 {
            next_state.x += 1;
        }
    }

    // Down
    else if action == 3 {
        // State 15 - Trying to move out of bounds
        if (xs == 3) && (ys == 3){
            next_state.reward = 0.0;
        }
        else if ys < 3 {
            next_state.y += 1;
        }
    }

    return next_state;
}

// Takes - Policy, discount, and cutoff threshold.
// Returns - Calculated values for every state.
fn policy_evaluation(policy: [[[f64;4];4];4], gamma: f64, theta: f64) -> [[f64;4];4]{
    let mut state_values: [[f64;4];4] = [[0.0;4];4];
    let (mut value, mut max_change, mut next_state_value_pair);

    loop {
        max_change = 0.0; // reset the change
        for ym in 0..4{
            for xm in 0..4{
                value = 0.0; // reset the value
                for m in 0..4{
                    // given a state and action return the next state and reword
                    next_state_value_pair = get_next_state(xm, ym, m);
                    // (probability that the action is chosen under the given policy)
                    // * ((reward) + (discount * value of next state))
                    value += policy[m][ym][xm] * (next_state_value_pair.reward +
                        (gamma * state_values[next_state_value_pair.y][next_state_value_pair.x]));
                }
                if (state_values[ym][xm].abs() - value.abs()).abs() > max_change {
                    max_change = (state_values[ym][xm].abs() - value.abs()).abs();
                }
                state_values[ym][xm] = value;
            }
        }

        if max_change < theta{ return state_values; } //
    }
}

// Takes - Policy and values for that policy.
// Returns - Improved policy.
fn policy_improvement( old_policy: [[[f64;4];4];4], values: [[f64;4];4])->PolicyStabilityPair{
    let mut updates = PolicyStabilityPair{pi: [[[0.0;4];4];4], stable: true};
    let (mut best_action, mut action_value, mut old_action, mut old_action_value, mut next_state_value_pair);

    for ys in 0..4{
        for xs in 0..4{
            action_value = f64::NEG_INFINITY;
            old_action_value = old_policy[0][ys][xs];
            best_action = 0;
            old_action = 0;

            for a in 0..4{
                next_state_value_pair = get_next_state(xs, ys, a);
                if values[next_state_value_pair.y][next_state_value_pair.x] > action_value{
                    action_value = values[next_state_value_pair.y][next_state_value_pair.x];
                    best_action = a;
                }
                if old_policy[a][ys][xs] > old_action_value{
                    old_action = a;
                }
            }
            updates.pi[best_action][ys][xs] = 1.0;
            if old_action != best_action{
                updates.stable = false;
            }
        }
    }
    return updates;
}

fn calculate_stochastic_policy(values: [[f64;4];4])->[[[f64;4];4];4]{
    let mut s_pi = [[[0.0;4];4];4];
    let (mut action_value, mut next_state_value_pair, mut sum);
    let mut actions = [0,0,0,0];


    for i in 0..4{
        for j in 0..4{
            action_value = f64::NEG_INFINITY;
            sum = 0.0;

            for a in 0..4{
                actions[a] = 0;
                next_state_value_pair = get_next_state(j, i, a);
                if values[next_state_value_pair.y][next_state_value_pair.x] > action_value{
                    action_value = values[next_state_value_pair.y][next_state_value_pair.x];
                }
            }
            for a in 0..4{
                next_state_value_pair = get_next_state(j, i, a);
                if values[next_state_value_pair.y][next_state_value_pair.x] >= action_value{
                    actions[a] = 1;
                    sum += 1.0;
                }
            }
            for a in 0..4{
                if actions[a] == 1{
                    s_pi[a][i][j] = 1.0/sum;
                }
            }
        }
    }
    return s_pi;
}

fn calculate_number_deterministic_policies(values: [[f64;4];4])->i32{
    let (mut action_value, mut next_state_value_pair, mut sum);
    let mut policies = 1;

    for i in 0..4{
        for j in 0..4{
            action_value = f64::NEG_INFINITY;
            sum = 0;
            for a in 0..4{
                next_state_value_pair = get_next_state(j, i, a);
                if values[next_state_value_pair.y][next_state_value_pair.x] > action_value{
                    action_value = values[next_state_value_pair.y][next_state_value_pair.x];
                }
            }
            for a in 0..4{
                next_state_value_pair = get_next_state(j, i, a);
                if values[next_state_value_pair.y][next_state_value_pair.x] >= action_value{
                    sum += 1;
                }
            }
            policies = policies*sum;
        }
    }

    return policies;
}

fn main() {
    let gamma = 0.95; // Discount value
    let theta = 0.00000001; // The threshold to determine convergence.
    let mut d_policy = PolicyStabilityPair{pi: [[[0.25;4];4];4], stable: false};
    let mut d_values:[[f64;4];4] = [[0.0;4];4];
    let mut s_policy = PolicyStabilityPair{pi: [[[0.25;4];4];4], stable: false};
    let s_values:[[f64;4];4];

    while !d_policy.stable { // This is the policy iteration.
        d_values = policy_evaluation(d_policy.pi, gamma, theta);
        d_policy = policy_improvement(d_policy.pi, d_values);
    }
    
    s_policy.pi = d_policy.pi;
    s_values = policy_evaluation(s_policy.pi, gamma, theta);
    s_policy.pi = calculate_stochastic_policy(s_values);

    print_formatted_results(d_policy.pi, d_values, s_policy.pi, s_values);
}